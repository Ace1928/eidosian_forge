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
@registry.architectures('spacy.Tok2VecListener.v1')
def tok2vec_listener_v1(width: int, upstream: str='*'):
    tok2vec = Tok2VecListener(upstream_name=upstream, width=width)
    return tok2vec