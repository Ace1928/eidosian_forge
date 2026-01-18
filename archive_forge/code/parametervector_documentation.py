from uuid import uuid4, UUID
from .parameter import Parameter
Resize the parameter vector.

        If necessary, new elements are generated. If length is smaller than before, the
        previous elements are cached and not re-generated if the vector is enlarged again.
        This is to ensure that the parameter instances do not change.
        