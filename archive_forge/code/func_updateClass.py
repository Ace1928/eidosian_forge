from __future__ import print_function
import gc
import inspect
import os
import sys
import traceback
import types
from .debug import printExc
def updateClass(old, new, debug):
    refs = gc.get_referrers(old)
    for ref in refs:
        try:
            if isinstance(ref, old) and ref.__class__ is old:
                ref.__class__ = new
                if debug:
                    print('    Changed class for %s' % safeStr(ref))
            elif inspect.isclass(ref) and issubclass(ref, old) and (old in ref.__bases__):
                ind = ref.__bases__.index(old)
                newBases = ref.__bases__[:ind] + (new, old) + ref.__bases__[ind + 1:]
                try:
                    ref.__bases__ = newBases
                except TypeError:
                    print('    Error setting bases for class %s' % ref)
                    print('        old bases: %s' % repr(ref.__bases__))
                    print('        new bases: %s' % repr(newBases))
                    raise
                if debug:
                    print('    Changed superclass for %s' % safeStr(ref))
        except Exception:
            print('Error updating reference (%s) for class change (%s -> %s)' % (safeStr(ref), safeStr(old), safeStr(new)))
            raise
    for attr in dir(old):
        oa = getattr(old, attr)
        if inspect.isfunction(oa) or inspect.ismethod(oa):
            try:
                na = getattr(new, attr)
            except AttributeError:
                if debug:
                    print('    Skipping method update for %s; new class does not have this attribute' % attr)
                continue
            ofunc = getattr(oa, '__func__', oa)
            nfunc = getattr(na, '__func__', na)
            if ofunc is not nfunc:
                depth = updateFunction(ofunc, nfunc, debug)
                if not hasattr(nfunc, '__previous_reload_method__'):
                    nfunc.__previous_reload_method__ = oa
                if debug:
                    extra = ''
                    if depth > 0:
                        extra = ' (and %d previous versions)' % depth
                    print('    Updating method %s%s' % (attr, extra))
    for attr in dir(new):
        if attr == '__previous_reload_version__':
            continue
        if not hasattr(old, attr):
            if debug:
                print('    Adding missing attribute %s' % attr)
            setattr(old, attr, getattr(new, attr))
    if hasattr(old, '__previous_reload_version__'):
        updateClass(old.__previous_reload_version__, new, debug)