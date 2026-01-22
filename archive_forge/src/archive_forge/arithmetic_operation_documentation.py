import abc
import itertools
from typing import Union, Iterable, List, Sequence, cast, Tuple, TYPE_CHECKING
from typing_extensions import Self
import numpy as np
from cirq.ops.raw_types import Gate
Returns the result of the gate operating on classical values.

        For example, an addition takes two values (the target and the source),
        adds the source into the target, then returns the target and source
        as the new register values.

        The `apply` method is permitted to be sloppy in three ways:

        1. The `apply` method is permitted to return values that have more bits
            than the registers they will be stored into. The extra bits are
            simply dropped. For example, if the value 5 is returned for a 2
            qubit register then 5 % 2**2 = 1 will be used instead. Negative
            values are also permitted. For example, for a 3 qubit register the
            value -2 becomes -2 % 2**3 = 6.
        2. When the value of the last `k` registers is not changed by the
            gate, the `apply` method is permitted to omit these values
            from the result. That is to say, when the length of the output is
            less than the length of the input, it is padded up to the intended
            length by copying from the same position in the input.
        3. When only the first register's value changes, the `apply` method is
            permitted to return an `int` instead of a sequence of ints.

        The `apply` method *must* be reversible. Otherwise the gate will
        not be unitary, and incorrect behavior will result.

        Examples:

            A fully detailed adder:

            ```
            def apply(self, target, offset):
                return (target + offset) % 2**len(self.target_register), offset
            ```

            The same adder, with less boilerplate due to the details being
            handled by the `ArithmeticGate` class:

            ```
            def apply(self, target, offset):
                return target + offset
            ```
        