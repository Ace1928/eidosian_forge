import torch.nn as nn
import torch.nn.functional as F
import parlai.core.torch_generator_agent as tga
class Seq2seqAgent(tga.TorchGeneratorAgent):
    """
    Example agent.

    Implements the interface for TorchGeneratorAgent. The minimum requirement is that it
    implements ``build_model``, but we will want to include additional command line
    parameters.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add CLI arguments.
        """
        super(Seq2seqAgent, cls).add_cmdline_args(argparser)
        group = argparser.add_argument_group('Example TGA Agent')
        group.add_argument('-hid', '--hidden-size', type=int, default=1024, help='Hidden size.')

    def build_model(self):
        """
        Construct the model.
        """
        model = ExampleModel(self.dict, self.opt['hidden_size'])
        self._copy_embeddings(model.embeddings.weight, self.opt['embedding_type'])
        return model