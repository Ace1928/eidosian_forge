import re
import torch._C as C

    Returns the computed dispatch table(str). Note this only include
    runtime keys, registrations to alias keys have been decoded to their
    mapped runtime keys.
    