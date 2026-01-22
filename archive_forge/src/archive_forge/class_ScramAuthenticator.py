import asyncio
import collections
import base64
import functools
import hashlib
import hmac
import logging
import random
import socket
import struct
import sys
import time
import traceback
import uuid
import warnings
import weakref
import async_timeout
import aiokafka.errors as Errors
from aiokafka.abc import AbstractTokenProvider
from aiokafka.protocol.api import RequestHeader
from aiokafka.protocol.admin import (
from aiokafka.protocol.commit import (
from aiokafka.util import create_future, create_task, get_running_loop, wait_for
class ScramAuthenticator(BaseSaslAuthenticator):
    MECHANISMS = {'SCRAM-SHA-256': hashlib.sha256, 'SCRAM-SHA-512': hashlib.sha512}

    def __init__(self, *, loop, sasl_plain_password, sasl_plain_username, sasl_mechanism):
        self._loop = loop
        self._nonce = str(uuid.uuid4()).replace('-', '')
        self._auth_message = ''
        self._salted_password = None
        self._sasl_plain_username = sasl_plain_username
        self._sasl_plain_password = sasl_plain_password.encode('utf-8')
        self._hashfunc = self.MECHANISMS[sasl_mechanism]
        self._hashname = ''.join(sasl_mechanism.lower().split('-')[1:3])
        self._stored_key = None
        self._client_key = None
        self._client_signature = None
        self._client_proof = None
        self._server_key = None
        self._server_signature = None
        self._authenticator = self.authenticator_scram()

    def first_message(self):
        client_first_bare = f'n={self._sasl_plain_username},r={self._nonce}'
        self._auth_message += client_first_bare
        return 'n,,' + client_first_bare

    def process_server_first_message(self, server_first):
        self._auth_message += ',' + server_first
        params = dict((pair.split('=', 1) for pair in server_first.split(',')))
        server_nonce = params['r']
        if not server_nonce.startswith(self._nonce):
            raise ValueError('Server nonce, did not start with client nonce!')
        self._nonce = server_nonce
        self._auth_message += ',c=biws,r=' + self._nonce
        salt = base64.b64decode(params['s'].encode('utf-8'))
        iterations = int(params['i'])
        self.create_salted_password(salt, iterations)
        self._client_key = self.hmac(self._salted_password, b'Client Key')
        self._stored_key = self._hashfunc(self._client_key).digest()
        self._client_signature = self.hmac(self._stored_key, self._auth_message.encode('utf-8'))
        self._client_proof = ScramAuthenticator._xor_bytes(self._client_key, self._client_signature)
        self._server_key = self.hmac(self._salted_password, b'Server Key')
        self._server_signature = self.hmac(self._server_key, self._auth_message.encode('utf-8'))

    def final_message(self):
        client_proof = base64.b64encode(self._client_proof).decode('utf-8')
        return f'c=biws,r={self._nonce},p={client_proof}'

    def process_server_final_message(self, server_final):
        params = dict((pair.split('=', 1) for pair in server_final.split(',')))
        if self._server_signature != base64.b64decode(params['v'].encode('utf-8')):
            raise ValueError('Server sent wrong signature!')

    def authenticator_scram(self):
        client_first = self.first_message().encode('utf-8')
        server_first = (yield (client_first, True))
        self.process_server_first_message(server_first.decode('utf-8'))
        client_final = self.final_message().encode('utf-8')
        server_final = (yield (client_final, True))
        self.process_server_final_message(server_final.decode('utf-8'))

    def hmac(self, key, msg):
        return hmac.new(key, msg, digestmod=self._hashfunc).digest()

    def create_salted_password(self, salt, iterations):
        self._salted_password = hashlib.pbkdf2_hmac(self._hashname, self._sasl_plain_password, salt, iterations)

    @staticmethod
    def _xor_bytes(left, right):
        return bytes((lb ^ rb for lb, rb in zip(left, right)))