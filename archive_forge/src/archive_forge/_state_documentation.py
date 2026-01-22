import os
import weakref
import torch
Stores whether the JIT is enabled or not.

    This is just a wrapper for a bool, so that we get reference semantics
    