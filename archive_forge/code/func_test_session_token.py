import unittest
import os
from boto.exception import BotoServerError
from boto.sts.connection import STSConnection
from boto.sts.credentials import Credentials
from boto.s3.connection import S3Connection
def test_session_token(self):
    print('--- running Session Token tests ---')
    c = STSConnection()
    token = c.get_session_token()
    token.save('token.json')
    token_copy = Credentials.load('token.json')
    assert token_copy.access_key == token.access_key
    assert token_copy.secret_key == token.secret_key
    assert token_copy.session_token == token.session_token
    assert token_copy.expiration == token.expiration
    assert token_copy.request_id == token.request_id
    os.unlink('token.json')
    assert not token.is_expired()
    s3 = S3Connection(aws_access_key_id=token.access_key, aws_secret_access_key=token.secret_key, security_token=token.session_token)
    buckets = s3.get_all_buckets()
    print('--- tests completed ---')