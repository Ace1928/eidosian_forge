from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def adjustSVGTagNames(self, token):
    replacements = {'altglyph': 'altGlyph', 'altglyphdef': 'altGlyphDef', 'altglyphitem': 'altGlyphItem', 'animatecolor': 'animateColor', 'animatemotion': 'animateMotion', 'animatetransform': 'animateTransform', 'clippath': 'clipPath', 'feblend': 'feBlend', 'fecolormatrix': 'feColorMatrix', 'fecomponenttransfer': 'feComponentTransfer', 'fecomposite': 'feComposite', 'feconvolvematrix': 'feConvolveMatrix', 'fediffuselighting': 'feDiffuseLighting', 'fedisplacementmap': 'feDisplacementMap', 'fedistantlight': 'feDistantLight', 'feflood': 'feFlood', 'fefunca': 'feFuncA', 'fefuncb': 'feFuncB', 'fefuncg': 'feFuncG', 'fefuncr': 'feFuncR', 'fegaussianblur': 'feGaussianBlur', 'feimage': 'feImage', 'femerge': 'feMerge', 'femergenode': 'feMergeNode', 'femorphology': 'feMorphology', 'feoffset': 'feOffset', 'fepointlight': 'fePointLight', 'fespecularlighting': 'feSpecularLighting', 'fespotlight': 'feSpotLight', 'fetile': 'feTile', 'feturbulence': 'feTurbulence', 'foreignobject': 'foreignObject', 'glyphref': 'glyphRef', 'lineargradient': 'linearGradient', 'radialgradient': 'radialGradient', 'textpath': 'textPath'}
    if token['name'] in replacements:
        token['name'] = replacements[token['name']]