import unittest
from traits.api import HasTraits, Str, Instance, Any
 Tests that demonstrate that trait events are delivered in LIFO
    order rather than FIFO order.

    Baz receives the "effect" event before it receives the "cause" event.
    