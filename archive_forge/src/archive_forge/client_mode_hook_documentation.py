import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
Runs a preregistered actor class on the ray client

    The common case for this decorator is for instantiating an ActorClass
    transparently as a ClientActorClass. This happens in circumstances where
    the ActorClass is declared early, in a library and only then is Ray used in
    client mode -- necessitating a conversion.
    