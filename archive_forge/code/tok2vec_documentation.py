from typing import List, Optional, Union, cast
from thinc.api import (
from thinc.types import Floats2d, Ints1d, Ints2d, Ragged
from ...attrs import intify_attr
from ...errors import Errors
from ...ml import _character_embed
from ...pipeline.tok2vec import Tok2VecListener
from ...tokens import Doc
from ...util import registry
from ..featureextractor import FeatureExtractor
from ..staticvectors import StaticVectors
Encode context using bidirectonal LSTM layers. Requires PyTorch.

    width (int): The input and output width. These are required to be the same,
        to allow residual connections. This value will be determined by the
        width of the inputs. Recommended values are between 64 and 300.
    depth (int): The number of recurrent layers.
    dropout (float): Creates a Dropout layer on the outputs of each LSTM layer
        except the last layer. Set to 0 to disable this functionality.
    