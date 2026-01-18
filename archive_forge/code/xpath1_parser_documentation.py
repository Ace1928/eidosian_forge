import re
from abc import ABCMeta
from typing import cast, Any, ClassVar, Dict, MutableMapping, \
from ..exceptions import MissingContextError, ElementPathValueError, \
from ..datatypes import QName
from ..tdop import Token, Parser
from ..namespaces import NamespacesType, XML_NAMESPACE, XSD_NAMESPACE, \
from ..sequence_types import match_sequence_type
from ..schema_proxy import AbstractSchemaProxy
from ..xpath_tokens import NargsType, XPathToken, XPathAxis, XPathFunction, \

        Returns an XPathFunction object suitable for stand-alone usage.

        :param name: the name of the function.
        :param arity: the arity of the function object, must be compatible         with the signature of the XPath function.
        :param context: an optional context to bound to the function.
        