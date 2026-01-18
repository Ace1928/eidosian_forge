import sys
import re
import io
import importlib
from typing import cast, Any, Counter, Iterator, Optional, MutableMapping, \
from .protocols import ElementProtocol, DocumentProtocol
import xml.etree.ElementTree as ElementTree
import xml.etree.ElementTree as PyElementTree  # noqa
import xml.etree  # noqa
def unparsed_entity_declaration(self, entity_name, base, system_id, public_id, notation_name):
    raise PyElementTree.ParseError('Unparsed entities are forbidden (entity_name={!r})'.format(entity_name))