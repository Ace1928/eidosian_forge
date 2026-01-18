from fontTools.misc.fixedTools import fixedToFloat as fi2fl, floatToFixed as fl2fi
from fontTools.misc.textTools import tobytes, tostr
import struct
import re
sstruct.py -- SuperStruct

Higher level layer on top of the struct module, enabling to
bind names to struct elements. The interface is similar to
struct, except the objects passed and returned are not tuples
(or argument lists), but dictionaries or instances.

Just like struct, we use fmt strings to describe a data
structure, except we use one line per element. Lines are
separated by newlines or semi-colons. Each line contains
either one of the special struct characters ('@', '=', '<',
'>' or '!') or a 'name:formatchar' combo (eg. 'myFloat:f').
Repetitions, like the struct module offers them are not useful
in this context, except for fixed length strings  (eg. 'myInt:5h'
is not allowed but 'myString:5s' is). The 'x' fmt character
(pad byte) is treated as 'special', since it is by definition
anonymous. Extra whitespace is allowed everywhere.

The sstruct module offers one feature that the "normal" struct
module doesn't: support for fixed point numbers. These are spelled
as "n.mF", where n is the number of bits before the point, and m
the number of bits after the point. Fixed point numbers get
converted to floats.

pack(fmt, object):
	'object' is either a dictionary or an instance (or actually
	anything that has a __dict__ attribute). If it is a dictionary,
	its keys are used for names. If it is an instance, it's
	attributes are used to grab struct elements from. Returns
	a string containing the data.

unpack(fmt, data, object=None)
	If 'object' is omitted (or None), a new dictionary will be
	returned. If 'object' is a dictionary, it will be used to add
	struct elements to. If it is an instance (or in fact anything
	that has a __dict__ attribute), an attribute will be added for
	each struct element. In the latter two cases, 'object' itself
	is returned.

unpack2(fmt, data, object=None)
	Convenience function. Same as unpack, except data may be longer
	than needed. The returned value is a tuple: (object, leftoverdata).

calcsize(fmt)
	like struct.calcsize(), but uses our own fmt strings:
	it returns the size of the data in bytes.
