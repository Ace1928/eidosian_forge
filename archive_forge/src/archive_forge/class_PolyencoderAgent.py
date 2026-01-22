from typing import Any, Dict, Optional, Tuple
import torch
from parlai.core.opt import Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .biencoder import AddLabelFixedCandsTRA
from .modules import (
from .transformer import TransformerRankerAgent
class PolyencoderAgent(TorchRankerAgent):
    """
    Poly-encoder Agent.

    Equivalent of bert_ranker/polyencoder and biencoder_multiple_output but does not
    rely on an external library (hugging face).
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        TransformerRankerAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Polyencoder Arguments')
        agent.add_argument('--polyencoder-type', type=str, default='codes', choices=['codes', 'n_first'], help='Type of polyencoder, either we computevectors using codes + attention, or we simply take the first N vectors.', recommended='codes')
        agent.add_argument('--poly-n-codes', type=int, default=64, help='number of vectors used to represent the contextin the case of n_first, those are the numberof vectors that are considered.', recommended=64)
        agent.add_argument('--poly-attention-type', type=str, default='basic', choices=['basic', 'sqrt', 'multihead'], help='Type of the top aggregation layer of the poly-encoder (where the candidate representation isthe key)', recommended='basic')
        agent.add_argument('--poly-attention-num-heads', type=int, default=4, help='In case poly-attention-type is multihead, specify the number of heads')
        agent.add_argument('--codes-attention-type', type=str, default='basic', choices=['basic', 'sqrt', 'multihead'], help='Type ', recommended='basic')
        agent.add_argument('--codes-attention-num-heads', type=int, default=4, help='In case codes-attention-type is multihead, specify the number of heads')
        return agent

    @classmethod
    def upgrade_opt(cls, opt_from_disk: Opt):
        opt_from_disk = super(PolyencoderAgent, cls).upgrade_opt(opt_from_disk)
        polyencoder_attention_keys_value = opt_from_disk.get('polyencoder_attention_keys')
        if polyencoder_attention_keys_value is not None:
            if polyencoder_attention_keys_value == 'context':
                del opt_from_disk['polyencoder_attention_keys']
            else:
                raise NotImplementedError('This --polyencoder-attention-keys mode (found in commit 06f0d9f) is no longer supported!')
        return opt_from_disk

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)
        if self.use_cuda:
            self.rank_loss.cuda()

    def build_model(self, states=None):
        """
        Return built model.
        """
        return PolyEncoderModule(self.opt, self.dict, self.NULL_IDX)

    def vectorize(self, *args, **kwargs):
        """
        Add the start and end token to the labels.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = super().vectorize(*args, **kwargs)
        return obs

    def _set_text_vec(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        obs = super()._set_text_vec(*args, **kwargs)
        if 'text_vec' in obs and 'added_start_end_tokens' not in obs:
            obs.force_set('text_vec', self._add_start_end_tokens(obs['text_vec'], True, True))
            obs['added_start_end_tokens'] = True
        return obs

    def vectorize_fixed_candidates(self, *args, **kwargs):
        """
        Vectorize fixed candidates.

        Override to add start and end token when computing the candidate encodings in
        interactive mode.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        return super().vectorize_fixed_candidates(*args, **kwargs)

    def _make_candidate_encs(self, vecs):
        """
        Make candidate encs.

        The polyencoder module expects cand vecs to be 3D while torch_ranker_agent
        expects it to be 2D. This requires a little adjustment (used in interactive mode
        only)
        """
        rep = super()._make_candidate_encs(vecs)
        return rep.transpose(0, 1).contiguous()

    def encode_candidates(self, padded_cands):
        """
        Encode candidates.
        """
        padded_cands = padded_cands.unsqueeze(1)
        _, _, cand_rep = self.model(cand_tokens=padded_cands)
        return cand_rep

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        Score candidates.

        The Poly-encoder encodes the candidate and context independently. Then, the
        model applies additional attention before ultimately scoring a candidate.
        """
        bsz = self._get_batch_size(batch)
        ctxt_rep, ctxt_rep_mask, _ = self.model(**self._model_context_input(batch))
        if cand_encs is not None:
            if bsz == 1:
                cand_rep = cand_encs
            else:
                cand_rep = cand_encs.expand(bsz, cand_encs.size(1), -1)
        elif len(cand_vecs.shape) == 3:
            _, _, cand_rep = self.model(cand_tokens=cand_vecs)
        elif len(cand_vecs.shape) == 2:
            _, _, cand_rep = self.model(cand_tokens=cand_vecs.unsqueeze(1))
            num_cands = cand_rep.size(0)
            cand_rep = cand_rep.expand(num_cands, bsz, -1).transpose(0, 1).contiguous()
        scores = self.model(ctxt_rep=ctxt_rep, ctxt_rep_mask=ctxt_rep_mask, cand_rep=cand_rep)
        return scores

    def _get_batch_size(self, batch) -> int:
        """
        Return the size of the batch.

        Can be overridden by subclasses that do not always have text input.
        """
        return batch.text_vec.size(0)

    def _model_context_input(self, batch) -> Dict[str, Any]:
        """
        Create the input context value for the model.

        Must return a dictionary.  This will be passed directly into the model via
        `**kwargs`, i.e.,

        >>> model(**_model_context_input(batch))

        This is intentionally overridable so that richer models can pass additional
        inputs.
        """
        return {'ctxt_tokens': batch.text_vec}

    def load_state_dict(self, state_dict):
        """
        Override to account for codes.
        """
        if self.model.type == 'codes' and 'codes' not in state_dict:
            state_dict['codes'] = self.model.codes
        super().load_state_dict(state_dict)