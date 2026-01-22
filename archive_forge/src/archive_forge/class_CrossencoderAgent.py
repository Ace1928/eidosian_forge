from .modules import TransformerEncoder
from .modules import get_n_positions_from_options
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .transformer import TransformerRankerAgent
from parlai.utils.torch import concat_without_padding
import torch
class CrossencoderAgent(TorchRankerAgent):
    """
    Equivalent of bert_ranker/crossencoder but does not rely on an external library
    (hugging face).
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        TransformerRankerAgent.add_cmdline_args(argparser)
        argparser.set_defaults(encode_candidate_vecs=False)
        return argparser

    def build_model(self, states=None):
        return CrossEncoderModule(self.opt, self.dict, self.NULL_IDX)

    def vectorize(self, *args, **kwargs):
        """
        Add the start and end token to the text.
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

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        if cand_encs is not None:
            raise Exception('Candidate pre-computation is impossible on the crossencoder')
        num_cands_per_sample = cand_vecs.size(1)
        bsz = cand_vecs.size(0)
        text_idx = batch.text_vec.unsqueeze(1).expand(-1, num_cands_per_sample, -1).contiguous().view(num_cands_per_sample * bsz, -1)
        cand_idx = cand_vecs.view(num_cands_per_sample * bsz, -1)
        tokens, segments = concat_without_padding(text_idx, cand_idx, self.use_cuda, self.NULL_IDX)
        scores = self.model(tokens, segments)
        scores = scores.view(bsz, num_cands_per_sample)
        return scores