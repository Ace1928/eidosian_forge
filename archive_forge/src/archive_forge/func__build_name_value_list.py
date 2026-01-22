import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
def _build_name_value_list(self, params, attributes, replace=False, label='Attribute'):
    keys = sorted(attributes.keys())
    i = 1
    for key in keys:
        value = attributes[key]
        if isinstance(value, list):
            for v in value:
                params['%s.%d.Name' % (label, i)] = key
                if self.converter:
                    v = self.converter.encode(v)
                params['%s.%d.Value' % (label, i)] = v
                if replace:
                    params['%s.%d.Replace' % (label, i)] = 'true'
                i += 1
        else:
            params['%s.%d.Name' % (label, i)] = key
            if self.converter:
                value = self.converter.encode(value)
            params['%s.%d.Value' % (label, i)] = value
            if replace:
                params['%s.%d.Replace' % (label, i)] = 'true'
        i += 1