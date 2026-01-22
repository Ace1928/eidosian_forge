from collections import defaultdict
from copy import deepcopy
import datetime
import io
import itertools
import logging
import os
import shutil
import tempfile
from urllib.parse import urlparse
import dateutil.parser
from prov import Error, serializers
from prov.constants import *
from prov.identifier import Identifier, QualifiedName, Namespace
class ProvRecord(object):
    """Base class for PROV records."""
    FORMAL_ATTRIBUTES = ()
    _prov_type = None
    'PROV type of record.'

    def __init__(self, bundle, identifier, attributes=None):
        """
        Constructor.

        :param bundle: Bundle for the PROV record.
        :param identifier: (Unique) identifier of the record.
        :param attributes: Attributes to associate with the record (default: None).
        """
        self._bundle = bundle
        self._identifier = identifier
        self._attributes = defaultdict(set)
        if attributes:
            self.add_attributes(attributes)

    def __hash__(self):
        return hash((self.get_type(), self._identifier, frozenset(self.attributes)))

    def copy(self):
        """
        Return an exact copy of this record.
        """
        return PROV_REC_CLS[self.get_type()](self._bundle, self.identifier, self.attributes)

    def get_type(self):
        """Returns the PROV type of the record."""
        return self._prov_type

    def get_asserted_types(self):
        """Returns the set of all asserted PROV types of this record."""
        return self._attributes[PROV_TYPE]

    def add_asserted_type(self, type_identifier):
        """
        Adds a PROV type assertion to the record.

        :param type_identifier: PROV namespace identifier to add.
        """
        self._attributes[PROV_TYPE].add(type_identifier)

    def get_attribute(self, attr_name):
        """
        Returns the attribute of the given name.

        :param attr_name: Name of the attribute.
        :return: Tuple (name, value)
        """
        attr_name = self._bundle.valid_qualified_name(attr_name)
        return self._attributes[attr_name]

    @property
    def identifier(self):
        """Record's identifier."""
        return self._identifier

    @property
    def attributes(self):
        """
        All record attributes.

        :return: List of tuples (name, value)
        """
        return [(attr_name, value) for attr_name, values in self._attributes.items() for value in values]

    @property
    def args(self):
        """
        All values of the record's formal attributes.

        :return: Tuple
        """
        return tuple((first(self._attributes[attr_name]) for attr_name in self.FORMAL_ATTRIBUTES))

    @property
    def formal_attributes(self):
        """
        All names and values of the record's formal attributes.

        :return: Tuple of tuples (name, value)
        """
        return tuple(((attr_name, first(self._attributes[attr_name])) for attr_name in self.FORMAL_ATTRIBUTES))

    @property
    def extra_attributes(self):
        """
        All names and values of the record's attributes that are not formal
        attributes.

        :return: Tuple of tuples (name, value)
        """
        return [(attr_name, attr_value) for attr_name, attr_value in self.attributes if attr_name not in self.FORMAL_ATTRIBUTES]

    @property
    def bundle(self):
        """
        Bundle of the record.

        :return: :py:class:`ProvBundle`
        """
        return self._bundle

    @property
    def label(self):
        """Identifying label of the record."""
        return first(self._attributes[PROV_LABEL]) if self._attributes[PROV_LABEL] else self._identifier

    @property
    def value(self):
        """Value of the record."""
        return self._attributes[PROV_VALUE]

    def _auto_literal_conversion(self, literal):
        if isinstance(literal, ProvRecord):
            literal = literal.identifier
        if isinstance(literal, str):
            return str(literal)
        elif isinstance(literal, QualifiedName):
            return self._bundle.valid_qualified_name(literal)
        elif isinstance(literal, Literal) and literal.has_no_langtag():
            if literal.datatype:
                value = parse_xsd_types(literal.value, literal.datatype)
            else:
                value = self._auto_literal_conversion(literal.value)
            if value is not None:
                return value
        return literal

    def add_attributes(self, attributes):
        """
        Add attributes to the record.

        :param attributes: Dictionary of attributes, with keys being qualified
            identifiers. Alternatively an iterable of tuples (key, value) with the
            keys satisfying the same condition.
        """
        if attributes:
            if isinstance(attributes, dict):
                attributes = attributes.items()
            if PROV_ATTR_COLLECTION in [_i[0] for _i in attributes]:
                is_collection = True
            else:
                is_collection = False
            for attr_name, original_value in attributes:
                if original_value is None:
                    continue
                attr = self._bundle.valid_qualified_name(attr_name)
                if attr is None:
                    raise ProvExceptionInvalidQualifiedName(attr_name)
                if attr in PROV_ATTRIBUTE_QNAMES:
                    qname = original_value.identifier if isinstance(original_value, ProvRecord) else original_value
                    value = self._bundle.valid_qualified_name(qname)
                elif attr in PROV_ATTRIBUTE_LITERALS:
                    value = original_value if isinstance(original_value, datetime.datetime) else parse_xsd_datetime(original_value)
                else:
                    value = self._auto_literal_conversion(original_value)
                if value is None:
                    raise ProvException('Invalid value for attribute %s: %s' % (attr, original_value))
                if not is_collection and attr in PROV_ATTRIBUTES and self._attributes[attr]:
                    existing_value = first(self._attributes[attr])
                    is_not_same_value = True
                    try:
                        is_not_same_value = value != existing_value
                    except TypeError:
                        pass
                    if is_not_same_value:
                        raise ProvException('Cannot have more than one value for attribute %s' % attr)
                    else:
                        continue
                self._attributes[attr].add(value)

    def __eq__(self, other):
        if not isinstance(other, ProvRecord):
            return False
        if self.get_type() != other.get_type():
            return False
        if self._identifier and (not self._identifier == other._identifier):
            return False
        return set(self.attributes) == set(other.attributes)

    def __str__(self):
        return self.get_provn()

    def get_provn(self):
        """
        Returns the PROV-N representation of the record.

        :return: String
        """
        items = []
        relation_id = ''
        if self._identifier:
            identifier = str(self._identifier)
            if self.is_element():
                items.append(identifier)
            else:
                relation_id = identifier + '; '
        for attr in self.FORMAL_ATTRIBUTES:
            if attr in self._attributes and self._attributes[attr]:
                value = first(self._attributes[attr])
                items.append(value.isoformat() if isinstance(value, datetime.datetime) else str(value))
            else:
                items.append('-')
        extra = []
        for attr in self._attributes:
            if attr not in self.FORMAL_ATTRIBUTES:
                for value in self._attributes[attr]:
                    try:
                        provn_represenation = value.provn_representation()
                    except AttributeError:
                        provn_represenation = encoding_provn_value(value)
                    extra.append('%s=%s' % (str(attr), provn_represenation))
        if extra:
            items.append('[%s]' % ', '.join(extra))
        prov_n = '%s(%s%s)' % (PROV_N_MAP[self.get_type()], relation_id, ', '.join(items))
        return prov_n

    def is_element(self):
        """
        True, if the record is an element, False otherwise.

        :return: bool
        """
        return False

    def is_relation(self):
        """
        True, if the record is a relation, False otherwise.

        :return: bool
        """
        return False