import datetime
from hashlib import sha1
import logging
from pymongo import MongoClient
import pymongo.errors
import pymongo.uri_parser
from saml2.eptid import Eptid
from saml2.ident import IdentDB
from saml2.ident import Unknown
from saml2.ident import code_binary
from saml2.mdie import from_dict
from saml2.mdie import to_dict
from saml2.mdstore import InMemoryMetaData
from saml2.mdstore import load_metadata_modules
from saml2.mdstore import metadata_modules
from saml2.s_utils import PolicyError
from saml2.saml import NAMEID_FORMAT_PERSISTENT
class MetadataMDB(InMemoryMetaData):

    def __init__(self, attrc, database='', collection=''):
        super().__init__(attrc)
        self.mdb = MDB(database, collection)
        self.mdb.primary_key = 'entity_id'

    def _ext_service(self, entity_id, typ, service, binding):
        try:
            srvs = self[entity_id][typ]
        except KeyError:
            return None
        if not srvs:
            return srvs
        res = []
        for srv in srvs:
            if 'extensions' in srv:
                for elem in srv['extensions']['extension_elements']:
                    if elem['__class__'] == service:
                        if elem['binding'] == binding:
                            res.append(elem)
        return res

    def load(self):
        pass

    def items(self):
        for key, item in self.mdb.items():
            yield (key, unprotect(item['entity_description']))

    def keys(self):
        return self.mdb.keys()

    def values(self):
        for key, item in self.mdb.items():
            yield unprotect(item['entity_description'])

    def __contains__(self, item):
        return item in self.mdb

    def __getitem__(self, item):
        res = self.mdb.get(item)
        if not res:
            raise KeyError(item)
        elif len(res) == 1:
            return unprotect(res[0]['entity_description'])
        else:
            raise CorruptDatabase(f'More then one document with key {item}')

    def bindings(self, entity_id, typ, service):
        pass