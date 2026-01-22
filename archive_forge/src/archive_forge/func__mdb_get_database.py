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
def _mdb_get_database(uri, **kwargs):
    """
    Helper-function to connect to MongoDB and return a database object.

    The `uri' argument should be either a full MongoDB connection URI string,
    or just a database name in which case a connection to the default mongo
    instance at mongodb://localhost:27017 will be made.

    Performs explicit authentication if a username is provided in a connection
    string URI, since PyMongo does not always seem to do that as promised.

    :params database: name as string or (uri, name)
    :returns: pymongo database object
    """
    if 'tz_aware' not in kwargs:
        kwargs['tz_aware'] = True
    connection_factory = MongoClient
    try:
        _parsed_uri = pymongo.uri_parser.parse_uri(uri)
    except pymongo.errors.InvalidURI:
        db_name = uri
        _conn = connection_factory()
    else:
        db_name = _parsed_uri.get('database', 'pysaml2')
        _conn = connection_factory(uri, **kwargs)
    _db = _conn[db_name]
    return _db