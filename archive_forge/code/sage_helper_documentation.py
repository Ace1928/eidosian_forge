import doctest

Helper code for dealing with additional functionality when Sage is
present.

Any method which works only in Sage should be decorated with
"@sage_method" and any doctests (in Sage methods or not) which should
be run only in Sage should be styled with input prompt "sage:" rather
than the usual ">>>".

Similarly, doctests which require SnapPy should be styled in a block
where the first non-whitespace character is | followed by a space.
