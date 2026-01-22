from ._base import *
import operator as op
class LazyHasher:
    hasher = CryptContext(schemes=['argon2'], deprecated='auto')

    @classmethod
    def create(cls, pass_string: str):
        return LazyHasher.hasher.hash(pass_string)

    @classmethod
    def verify(cls, pass_hash: str, pass_string: str):
        return LazyHasher.hasher.verify(pass_string, pass_hash)

    @classmethod
    def update(cls, old_hash: str, old_pass: str, new_pass: str, do_verify: bool=True):
        if do_verify and LazyHasher.verify(old_hash, old_pass) or not do_verify:
            return LazyHasher.create(new_pass)
        return None

    @classmethod
    def create_token(cls):
        return str(uuid4())