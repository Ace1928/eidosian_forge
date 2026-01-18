from importlib import import_module
import inspect
import io
import token
import tokenize
import traceback
from sphinx.ext.autodoc import ClassLevelDocumenter
from sphinx.util import logging
from traits.has_traits import MetaHasTraits
from traits.trait_type import TraitType
from traits.traits import generic_trait
 Add the directive header 'attribute' with the annotation
        option set to the trait definition.

        