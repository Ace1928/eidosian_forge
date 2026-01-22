from __future__ import unicode_literals
import sys
import datetime
from decimal import Decimal
import os
import logging
import hashlib
import warnings
from . import __author__, __copyright__, __license__, __version__
from .simplexml import SimpleXMLElement
import random
import string
from hashlib import sha1
class UsernameDigestToken(UsernameToken):
    """
    WebService Security extension to add a http digest credentials to xml request
    drift -> time difference from the server in seconds, needed for 'Created' header
    """

    def __init__(self, username='', password='', drift=0):
        self.username = username
        self.password = password
        self.drift = datetime.timedelta(seconds=drift)

    def preprocess(self, client, request, method, args, kwargs, headers, soap_uri):
        header = request('Header', ns=soap_uri)
        wsse = header.add_child('wsse:Security', ns=False)
        wsse['xmlns:wsse'] = WSSE_URI
        wsse['xmlns:wsu'] = WSU_URI
        usertoken = wsse.add_child('wsse:UsernameToken', ns=False)
        usertoken.add_child('wsse:Username', self.username, ns=False)
        created = (datetime.datetime.utcnow() + self.drift).isoformat() + 'Z'
        usertoken.add_child('wsu:Created', created, ns=False)
        nonce = randombytes(16)
        wssenonce = usertoken.add_child('wsse:Nonce', nonce.encode('base64')[:-1], ns=False)
        wssenonce['EncodingType'] = Base64Binary_URI
        sha1obj = sha1()
        sha1obj.update(nonce + created + self.password)
        digest = sha1obj.digest()
        password = usertoken.add_child('wsse:Password', digest.encode('base64')[:-1], ns=False)
        password['Type'] = PasswordDigest_URI