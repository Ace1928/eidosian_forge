import enum
Support exporting switch-case statements as proposed by
    https://github.com/openqasm/openqasm/pull/463 at `commit bfa787aa3078
    <https://github.com/openqasm/openqasm/pull/463/commits/bfa787aa3078>`__.

    These have the output format:

    .. code-block::

        switch (i) {
            case 0:
            case 1:
                x $0;
            break;

            case 2: {
                z $0;
            }
            break;

            default: {
                cx $0, $1;
            }
            break;
        }

    This differs from the syntax of the ``switch`` statement as stabilized.  If this flag is not
    passed, then the parser will instead output using the stabilized syntax, which would render the
    same example above as:

    .. code-block::

        switch (i) {
            case 0, 1 {
                x $0;
            }
            case 2 {
                z $0;
            }
            default {
                cx $0, $1;
            }
        }
    