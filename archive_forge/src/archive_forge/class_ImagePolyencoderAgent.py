from typing import Any, Dict
import torch
from parlai.agents.image_seq2seq.modules import ContextWithImageEncoder
from parlai.agents.transformer.modules import get_n_positions_from_options
from parlai.agents.transformer.polyencoder import PolyencoderAgent, PolyEncoderModule
from parlai.core.torch_agent import Batch
from parlai.core.torch_image_agent import TorchImageAgent
from parlai.utils.misc import warn_once
class ImagePolyencoderAgent(PolyencoderAgent, TorchImageAgent):
    """
    Poly-encoder Agent that ingests image features.

    Agent that allows encoding image features and adding or concatenating them to the
    context encoding.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        PolyencoderAgent.add_cmdline_args(argparser)
        TorchImageAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('ImagePolyencoder Args')
        agent.add_argument('--image-combination-mode', type=str, default='prepend', choices=['add', 'append', 'prepend'], help='How to combine image embedding (if used) with context embedding')
        agent.set_defaults(reduction_type=None)
        return agent

    def build_model(self, states=None):
        """
        Return built model.
        """
        return ImagePolyencoderModule(self.opt, self.dict, self.NULL_IDX)

    def batchify_image_features(self, batch: Batch) -> Batch:
        """
        Return the image features as a Tensor of the correct type.

        Fill in missing feature vectors. Here, we require image features to be saved in
        `batch` as a Tensor for passing through the image encoder. This is required for
        data_parallel.
        """
        bsz = self._get_batch_size(batch)
        if batch.image is None or len(batch.image) == 0:
            batch.image = [None] * bsz
        else:
            assert len(batch.image) == bsz
        processed_features_list = []
        processed_zero_features = self._process_image_features(torch.zeros((self.image_features_dim,)))
        for orig_features in batch.image:
            if isinstance(orig_features, torch.Tensor):
                processed_features_list.append(self._process_image_features(orig_features))
            else:
                if orig_features is not None:
                    warn_once('Unsupported image feature format. Image features will be ignored!')
                processed_features_list.append(processed_zero_features)
        batch.image = torch.stack(processed_features_list)
        return batch

    def _get_batch_size(self, batch) -> int:
        """
        Return the size of the batch.

        Use the size of the text vec if it exists; otherwise, use the length of the
        image feature list.
        """
        if batch.text_vec is not None:
            return batch.text_vec.size(0)
        else:
            return len(batch.image)

    def _model_context_input(self, batch) -> Dict[str, Any]:
        """
        Override PolyencoderAgent's context inputs into the model.
        """
        return {'ctxt_tokens': batch.text_vec, 'ctxt_image': batch.image}

    def load_state_dict(self, state_dict):
        """
        Override to account for weights used for image features.
        """
        for tensor in ['dummy_image_enc', 'ones_mask']:
            key = f'encoder_ctxt.{tensor}'
            if hasattr(self.model.encoder_ctxt, tensor) and key not in state_dict:
                state_dict[key] = getattr(self.model.encoder_ctxt, tensor)
        if hasattr(self.model.encoder_ctxt, 'image_encoder'):
            for layer_idx, layer in enumerate(self.model.encoder_ctxt.image_encoder):
                for tensor in ['weight', 'bias']:
                    key = f'encoder_ctxt.image_encoder.{layer_idx}.{tensor}'
                    if hasattr(layer, tensor) and key not in state_dict:
                        state_dict[key] = getattr(layer, tensor)
        super().load_state_dict(state_dict)