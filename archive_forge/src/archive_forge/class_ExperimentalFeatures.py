import enum
class ExperimentalFeatures(enum.Flag):
    """Flags for experimental features that the OpenQASM 3 exporter supports.

    These are experimental and are more liable to change, because the OpenQASM 3
    specification has not formally accepted them yet, so the syntax may not be finalized."""
    SWITCH_CASE_V1 = enum.auto()
    'Support exporting switch-case statements as proposed by\n    https://github.com/openqasm/openqasm/pull/463 at `commit bfa787aa3078\n    <https://github.com/openqasm/openqasm/pull/463/commits/bfa787aa3078>`__.\n\n    These have the output format:\n\n    .. code-block::\n\n        switch (i) {\n            case 0:\n            case 1:\n                x $0;\n            break;\n\n            case 2: {\n                z $0;\n            }\n            break;\n\n            default: {\n                cx $0, $1;\n            }\n            break;\n        }\n\n    This differs from the syntax of the ``switch`` statement as stabilized.  If this flag is not\n    passed, then the parser will instead output using the stabilized syntax, which would render the\n    same example above as:\n\n    .. code-block::\n\n        switch (i) {\n            case 0, 1 {\n                x $0;\n            }\n            case 2 {\n                z $0;\n            }\n            default {\n                cx $0, $1;\n            }\n        }\n    '