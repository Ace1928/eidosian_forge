from typing import Any, Dict, Optional
Checks that eval(repr(v)) == v.

    Args:
        value: A value whose repr should be evaluatable python
            code that produces an equivalent value.
        setup_code: Code that must be executed before the repr can be evaluated.
            Ideally this should just be a series of 'import' lines.
        global_vals: Pre-defined values that should be in the global scope when
            evaluating the repr.
        local_vals: Pre-defined values that should be in the local scope when
            evaluating the repr.

    Raises:
        AssertionError: If the assertion fails, or eval(repr(value)) raises an error.
    