import re
import sys
import inspect
import operator
import itertools
import collections
from inspect import getfullargspec
class FunctionMaker:
    """
    An object with the ability to create functions with a given signature.
    It has attributes name, doc, module, signature, defaults, dict, and
    methods update and make.
    """
    _compile_count = itertools.count()

    def __init__(self, func=None, name=None, signature=None, defaults=None, doc=None, module=None, funcdict=None):
        self.shortsignature = signature
        if func:
            self.name = func.__name__
            if self.name == '<lambda>':
                self.name = '_lambda_'
            self.doc = func.__doc__
            self.module = func.__module__
            if inspect.isfunction(func):
                argspec = getfullargspec(func)
                self.annotations = getattr(func, '__annotations__', {})
                for a in ('args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults'):
                    setattr(self, a, getattr(argspec, a))
                for i, arg in enumerate(self.args):
                    setattr(self, 'arg%d' % i, arg)
                allargs = list(self.args)
                allshortargs = list(self.args)
                if self.varargs:
                    allargs.append('*' + self.varargs)
                    allshortargs.append('*' + self.varargs)
                elif self.kwonlyargs:
                    allargs.append('*')
                for a in self.kwonlyargs:
                    allargs.append('%s=None' % a)
                    allshortargs.append(f'{a}={a}')
                if self.varkw:
                    allargs.append('**' + self.varkw)
                    allshortargs.append('**' + self.varkw)
                self.signature = ', '.join(allargs)
                self.shortsignature = ', '.join(allshortargs)
                self.dict = func.__dict__.copy()
        if name:
            self.name = name
        if signature is not None:
            self.signature = signature
        if defaults:
            self.defaults = defaults
        if doc:
            self.doc = doc
        if module:
            self.module = module
        if funcdict:
            self.dict = funcdict
        assert hasattr(self, 'name')
        if not hasattr(self, 'signature'):
            raise TypeError('You are decorating a non-function: %s' % func)

    def update(self, func, **kw):
        """Update the signature of func with the data in self"""
        func.__name__ = self.name
        func.__doc__ = getattr(self, 'doc', None)
        func.__dict__ = getattr(self, 'dict', {})
        func.__defaults__ = getattr(self, 'defaults', ())
        func.__kwdefaults__ = getattr(self, 'kwonlydefaults', None)
        func.__annotations__ = getattr(self, 'annotations', None)
        try:
            frame = sys._getframe(3)
        except AttributeError:
            callermodule = '?'
        else:
            callermodule = frame.f_globals.get('__name__', '?')
        func.__module__ = getattr(self, 'module', callermodule)
        func.__dict__.update(kw)

    def make(self, src_templ, evaldict=None, addsource=False, **attrs):
        """Make a new function from a given template and update the signature"""
        src = src_templ % vars(self)
        evaldict = evaldict or {}
        mo = DEF.match(src)
        if mo is None:
            raise SyntaxError('not a valid function template\n%s' % src)
        name = mo.group(1)
        names = set([name] + [arg.strip(' *') for arg in self.shortsignature.split(',')])
        for n in names:
            if n in ('_func_', '_call_'):
                raise NameError(f'{n} is overridden in\n{src}')
        if not src.endswith('\n'):
            src += '\n'
        filename = '<decorator-gen-%d>' % (next(self._compile_count),)
        try:
            code = compile(src, filename, 'single')
            exec(code, evaldict)
        except:
            print('Error in generated code:', file=sys.stderr)
            print(src, file=sys.stderr)
            raise
        func = evaldict[name]
        if addsource:
            attrs['__source__'] = src
        self.update(func, **attrs)
        return func

    @classmethod
    def create(cls, obj, body, evaldict, defaults=None, doc=None, module=None, addsource=True, **attrs):
        """
        Create a function from the strings name, signature, and body.
        evaldict is the evaluation dictionary. If addsource is true, an
        attribute __source__ is added to the result. The attributes attrs
        are added, if any.
        """
        if isinstance(obj, str):
            name, rest = obj.strip().split('(', 1)
            signature = rest[:-1]
            func = None
        else:
            name = None
            signature = None
            func = obj
        self = cls(func, name, signature, defaults, doc, module)
        ibody = '\n'.join(('    ' + line for line in body.splitlines()))
        return self.make('def %(name)s(%(signature)s):\n' + ibody, evaldict, addsource, **attrs)