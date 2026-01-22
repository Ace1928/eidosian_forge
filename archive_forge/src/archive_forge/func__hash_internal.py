from __future__ import annotations
import hashlib
import hmac
import os
import posixpath
import secrets
def _hash_internal(method: str, salt: str, password: str) -> tuple[str, str]:
    method, *args = method.split(':')
    salt = salt.encode('utf-8')
    password = password.encode('utf-8')
    if method == 'scrypt':
        if not args:
            n = 2 ** 15
            r = 8
            p = 1
        else:
            try:
                n, r, p = map(int, args)
            except ValueError:
                raise ValueError("'scrypt' takes 3 arguments.") from None
        maxmem = 132 * n * r * p
        return (hashlib.scrypt(password, salt=salt, n=n, r=r, p=p, maxmem=maxmem).hex(), f'scrypt:{n}:{r}:{p}')
    elif method == 'pbkdf2':
        len_args = len(args)
        if len_args == 0:
            hash_name = 'sha256'
            iterations = DEFAULT_PBKDF2_ITERATIONS
        elif len_args == 1:
            hash_name = args[0]
            iterations = DEFAULT_PBKDF2_ITERATIONS
        elif len_args == 2:
            hash_name = args[0]
            iterations = int(args[1])
        else:
            raise ValueError("'pbkdf2' takes 2 arguments.")
        return (hashlib.pbkdf2_hmac(hash_name, password, salt, iterations).hex(), f'pbkdf2:{hash_name}:{iterations}')
    else:
        raise ValueError(f"Invalid hash method '{method}'.")