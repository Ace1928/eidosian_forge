import warnings
from transformers import AutoTokenizer
from transformers.utils import is_torch_available
from transformers.utils.generic import ExplicitEnum
from ...processing_utils import ProcessorMixin
def wp_decode(self, sequences):
    """
        Convert a list of lists of word piece token ids into a list of strings by calling word piece tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of wp decoded sentences.
        """
    decode_strs = [seq.replace(' ', '') for seq in self.wp_tokenizer.batch_decode(sequences)]
    return decode_strs