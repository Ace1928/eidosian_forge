from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
class DataShapeParser:
    """A DataShape parser object."""

    def __init__(self, ds_str, sym):
        self.ds_str = ds_str
        self.sym = sym
        self.lex = lexer.lex(ds_str)
        self.tokens = []
        self.pos = -1
        self.end_pos = None
        self.advance_tok()

    def advance_tok(self):
        """Advances self.pos by one, if it is not already at the end."""
        if self.pos != self.end_pos:
            self.pos = self.pos + 1
            try:
                if self.pos >= len(self.tokens):
                    self.tokens.append(next(self.lex))
            except StopIteration:
                if len(self.tokens) > 0:
                    span = (self.tokens[self.pos - 1].span[1],) * 2
                else:
                    span = (0, 0)
                self.tokens.append(lexer.Token(None, None, span, None))
                self.end_pos = self.pos

    @property
    def tok(self):
        return self.tokens[self.pos]

    def raise_error(self, errmsg):
        raise error.DataShapeSyntaxError(self.tok.span[0], '<nofile>', self.ds_str, errmsg)

    def parse_homogeneous_list(self, parse_item, sep_tok_id, errmsg, trailing_sep=False):
        """
        <item>_list : <item> <SEP> <item>_list
                    | <item>

        Returns a list of <item>s, or None.
        """
        saved_pos = self.pos
        items = []
        item = True
        while item is not None:
            item = parse_item()
            if item is not None:
                items.append(item)
                if self.tok.id == sep_tok_id:
                    self.advance_tok()
                else:
                    return items
            elif len(items) > 0:
                if trailing_sep:
                    return items
                else:
                    self.raise_error(errmsg)
            else:
                self.pos = saved_pos
                return None

    def syntactic_sugar(self, symdict, name, dshapemsg, error_pos=None):
        """
        Looks up a symbol in the provided symbol table dictionary for
        syntactic sugar, raising a standard error message if the symbol
        is missing.

        Parameters
        ----------
        symdict : symbol table dictionary
            One of self.sym.dtype, self.sym.dim,
            self.sym.dtype_constr, or self.sym.dim_constr.
        name : str
            The name of the symbol to look up.
        dshapemsg : str
            The datashape construct this lookup is for, e.g.
            '{...} dtype constructor'.
        error_pos : int, optional
            The position in the token stream at which to flag the error.
        """
        entry = symdict.get(name)
        if entry is not None:
            return entry
        else:
            if error_pos is not None:
                self.pos = error_pos
            self.raise_error(('Symbol table missing "%s" ' + 'entry for %s') % (name, dshapemsg))

    def parse_datashape(self):
        """
        datashape : datashape_nooption
                  | QUESTIONMARK datashape_nooption
                  | EXCLAMATIONMARK datashape_nooption

        Returns a datashape object or None.
        """
        tok = self.tok
        constructors = {lexer.QUESTIONMARK: 'option'}
        if tok.id in constructors:
            self.advance_tok()
            saved_pos = self.pos
            ds = self.parse_datashape_nooption()
            if ds is not None:
                option = self.syntactic_sugar(self.sym.dtype_constr, constructors[tok.id], '%s dtype construction' % constructors[tok.id], saved_pos - 1)
                return coretypes.DataShape(option(ds))
        else:
            return self.parse_datashape_nooption()

    def parse_datashape_nooption(self):
        """
        datashape_nooption : dim ASTERISK datashape
                           | dtype

        Returns a datashape object or None.
        """
        saved_pos = self.pos
        dim = self.parse_dim()
        if dim is not None:
            if self.tok.id == lexer.ASTERISK:
                self.advance_tok()
                saved_pos = self.pos
                dshape = self.parse_datashape()
                if dshape is None:
                    self.pos = saved_pos
                    self.raise_error('Expected a dim or a dtype')
                return coretypes.DataShape(dim, *dshape.parameters)
        dtype = self.parse_dtype()
        if dtype:
            return coretypes.DataShape(dtype)
        else:
            return None

    def parse_dim(self):
        """
        dim : typevar
            | ellipsis_typevar
            | type
            | type_constr
            | INTEGER
            | ELLIPSIS
        typevar : NAME_UPPER
        ellipsis_typevar : NAME_UPPER ELLIPSIS
        type : NAME_LOWER
        type_constr : NAME_LOWER LBRACKET type_arg_list RBRACKET

        Returns a the dim object, or None.
        TODO: Support type constructors
        """
        saved_pos = self.pos
        tok = self.tok
        if tok.id == lexer.NAME_UPPER:
            val = tok.val
            self.advance_tok()
            if self.tok.id == lexer.ELLIPSIS:
                self.advance_tok()
                tconstr = self.syntactic_sugar(self.sym.dim_constr, 'ellipsis', 'TypeVar... dim constructor', saved_pos)
                return tconstr(val)
            elif self.tok.id == lexer.ASTERISK:
                tconstr = self.syntactic_sugar(self.sym.dim_constr, 'typevar', 'TypeVar dim constructor', saved_pos)
                return tconstr(val)
            else:
                self.pos = saved_pos
                return None
        elif tok.id == lexer.NAME_LOWER:
            name = tok.val
            self.advance_tok()
            if self.tok.id == lexer.LBRACKET:
                dim_constr = self.sym.dim_constr.get(name)
                if dim_constr is None:
                    self.pos = saved_pos
                    return None
                self.advance_tok()
                args = self.parse_type_arg_list()
                if self.tok.id == lexer.RBRACKET:
                    self.advance_tok()
                    raise NotImplementedError('dim type constructors not actually supported yet')
                else:
                    self.raise_error('Expected a closing "]"')
            else:
                dim = self.sym.dim.get(name)
                if dim is not None:
                    return dim
                else:
                    self.pos = saved_pos
                    return None
        elif tok.id == lexer.INTEGER:
            val = tok.val
            self.advance_tok()
            if self.tok.id != lexer.ASTERISK:
                self.pos = saved_pos
                return None
            tconstr = self.syntactic_sugar(self.sym.dim_constr, 'fixed', 'integer dimensions')
            return tconstr(val)
        elif tok.id == lexer.ELLIPSIS:
            self.advance_tok()
            dim = self.syntactic_sugar(self.sym.dim, 'ellipsis', '... dim', saved_pos)
            return dim
        else:
            return None

    def parse_dtype(self):
        """
        dtype : typevar
              | type
              | type_constr
              | struct_type
              | funcproto_or_tuple_type
        typevar : NAME_UPPER
        ellipsis_typevar : NAME_UPPER ELLIPSIS
        type : NAME_LOWER
        type_constr : NAME_LOWER LBRACKET type_arg_list RBRACKET
        struct_type : LBRACE ...
        funcproto_or_tuple_type : LPAREN ...

        Returns a the dtype object, or None.
        """
        saved_pos = self.pos
        tok = self.tok
        if tok.id == lexer.NAME_UPPER:
            val = tok.val
            self.advance_tok()
            tconstr = self.syntactic_sugar(self.sym.dtype_constr, 'typevar', 'TypeVar dtype constructor', saved_pos)
            return tconstr(val)
        elif tok.id == lexer.NAME_LOWER:
            name = tok.val
            self.advance_tok()
            if self.tok.id == lexer.LBRACKET:
                dtype_constr = self.sym.dtype_constr.get(name)
                if dtype_constr is None:
                    self.pos = saved_pos
                    return None
                self.advance_tok()
                args, kwargs = self.parse_type_arg_list()
                if self.tok.id == lexer.RBRACKET:
                    if len(args) == 0 and len(kwargs) == 0:
                        self.raise_error('Expected at least one type ' + 'constructor argument')
                    self.advance_tok()
                    return dtype_constr(*args, **kwargs)
                else:
                    self.raise_error('Invalid type constructor argument')
            else:
                dtype = self.sym.dtype.get(name)
                if dtype is not None:
                    return dtype
                else:
                    self.pos = saved_pos
                    return None
        elif tok.id == lexer.LBRACE:
            return self.parse_struct_type()
        elif tok.id == lexer.LPAREN:
            return self.parse_funcproto_or_tuple_type()
        else:
            return None

    def parse_type_arg_list(self):
        """
        type_arg_list : type_arg COMMA type_arg_list
                      | type_kwarg_list
                      | type_arg
        type_kwarg_list : type_kwarg COMMA type_kwarg_list
                        | type_kwarg

        Returns a tuple (args, kwargs), or (None, None).
        """
        args = []
        arg = True
        while arg is not None:
            arg = self.parse_type_arg()
            if arg is not None:
                if self.tok.id == lexer.COMMA:
                    self.advance_tok()
                    args.append(arg)
                else:
                    args.append(arg)
                    return (args, {})
            else:
                break
        kwargs = self.parse_homogeneous_list(self.parse_type_kwarg, lexer.COMMA, 'Expected another keyword argument, ' + 'positional arguments cannot follow ' + 'keyword arguments')
        return (args, dict(kwargs) if kwargs else {})

    def parse_type_arg(self):
        """
        type_arg : datashape
                 | INTEGER
                 | STRING
                 | BOOLEAN
                 | list_type_arg
        list_type_arg : LBRACKET RBRACKET
                      | LBRACKET datashape_list RBRACKET
                      | LBRACKET integer_list RBRACKET
                      | LBRACKET string_list RBRACKET

        Returns a type_arg value, or None.
        """
        ds = self.parse_datashape()
        if ds is not None:
            return ds
        if self.tok.id in [lexer.INTEGER, lexer.STRING, lexer.BOOLEAN]:
            val = self.tok.val
            self.advance_tok()
            return val
        elif self.tok.id == lexer.LBRACKET:
            self.advance_tok()
            val = self.parse_datashape_list()
            if val is None:
                val = self.parse_integer_list()
            if val is None:
                val = self.parse_string_list()
            if val is None:
                val = self.parse_boolean_list()
            if self.tok.id == lexer.RBRACKET:
                self.advance_tok()
                return [] if val is None else val
            elif val is None:
                self.raise_error('Expected a type constructor argument ' + 'or a closing "]"')
            else:
                self.raise_error('Expected a "," or a closing "]"')
        else:
            return None

    def parse_type_kwarg(self):
        """
        type_kwarg : NAME_LOWER EQUAL type_arg

        Returns a (name, type_arg) tuple, or None.
        """
        if self.tok.id != lexer.NAME_LOWER:
            return None
        saved_pos = self.pos
        name = self.tok.val
        self.advance_tok()
        if self.tok.id != lexer.EQUAL:
            self.pos = saved_pos
            return None
        self.advance_tok()
        arg = self.parse_type_arg()
        if arg is not None:
            return (name, arg)
        else:
            self.raise_error('Expected a type constructor argument')

    def parse_datashape_list(self):
        """
        datashape_list : datashape COMMA datashape_list
                       | datashape

        Returns a list of datashape type objects, or None.
        """
        return self.parse_homogeneous_list(self.parse_datashape, lexer.COMMA, 'Expected another datashape, ' + 'type constructor parameter ' + 'lists must have uniform type')

    def parse_integer(self):
        """
        integer : INTEGER
        """
        if self.tok.id == lexer.INTEGER:
            val = self.tok.val
            self.advance_tok()
            return val
        else:
            return None

    def parse_integer_list(self):
        """
        integer_list : INTEGER COMMA integer_list
                     | INTEGER

        Returns a list of integers, or None.
        """
        return self.parse_homogeneous_list(self.parse_integer, lexer.COMMA, 'Expected another integer, ' + 'type constructor parameter ' + 'lists must have uniform type')

    def parse_boolean(self):
        """
        boolean : BOOLEAN
        """
        if self.tok.id == lexer.BOOLEAN:
            val = self.tok.val
            self.advance_tok()
            return val
        else:
            return None

    def parse_boolean_list(self):
        """
        boolean_list : boolean COMMA boolean_list
                     | boolean

        Returns a list of booleans, or None.
        """
        return self.parse_homogeneous_list(self.parse_boolean, lexer.COMMA, 'Expected another boolean, ' + 'type constructor parameter ' + 'lists must have uniform type')

    def parse_string(self):
        """
        string : STRING
        """
        if self.tok.id == lexer.STRING:
            val = self.tok.val
            self.advance_tok()
            return val
        else:
            return None

    def parse_string_list(self):
        """
        string_list : STRING COMMA string_list
                    | STRING

        Returns a list of strings, or None.
        """
        return self.parse_homogeneous_list(self.parse_string, lexer.COMMA, 'Expected another string, ' + 'type constructor parameter ' + 'lists must have uniform type')

    def parse_struct_type(self):
        """
        struct_type : LBRACE struct_field_list RBRACE
                    | LBRACE struct_field_list COMMA RBRACE

        Returns a struct type, or None.
        """
        if self.tok.id != lexer.LBRACE:
            return None
        saved_pos = self.pos
        self.advance_tok()
        fields = self.parse_homogeneous_list(self.parse_struct_field, lexer.COMMA, 'Invalid field in struct', trailing_sep=True) or []
        if self.tok.id != lexer.RBRACE:
            self.raise_error('Invalid field in struct')
        self.advance_tok()
        names = [f[0] for f in fields]
        types = [f[1] for f in fields]
        tconstr = self.syntactic_sugar(self.sym.dtype_constr, 'struct', '{...} dtype constructor', saved_pos)
        return tconstr(names, types)

    def parse_struct_field(self):
        """
        struct_field : struct_field_name COLON datashape
        struct_field_name : NAME_LOWER
                          | NAME_UPPER
                          | NAME_OTHER
                          | STRING

        Returns a tuple (name, datashape object) or None
        """
        if self.tok.id not in [lexer.NAME_LOWER, lexer.NAME_UPPER, lexer.NAME_OTHER, lexer.STRING]:
            return None
        name = self.tok.val
        self.advance_tok()
        if self.tok.id != lexer.COLON:
            self.raise_error('Expected a ":" separating the field ' + 'name from its datashape')
        self.advance_tok()
        ds = self.parse_datashape()
        if ds is None:
            self.raise_error('Expected the datashape of the field')
        return (name, ds)

    def parse_funcproto_or_tuple_type(self):
        """
        funcproto_or_tuple_type : tuple_type RARROW datashape
                                | tuple_type
        tuple_type : LPAREN tuple_item_list RPAREN
                   | LPAREN tuple_item_list COMMA RPAREN
                   | LPAREN RPAREN
        tuple_item_list : datashape COMMA tuple_item_list
                        | datashape

        Returns a tuple type object, a function prototype, or None.
        """
        if self.tok.id != lexer.LPAREN:
            return None
        saved_pos = self.pos
        self.advance_tok()
        dshapes = self.parse_homogeneous_list(self.parse_datashape, lexer.COMMA, 'Invalid datashape in tuple', trailing_sep=True) or ()
        if self.tok.id != lexer.RPAREN:
            self.raise_error('Invalid datashape in tuple')
        self.advance_tok()
        if self.tok.id != lexer.RARROW:
            tconstr = self.syntactic_sugar(self.sym.dtype_constr, 'tuple', '(...) dtype constructor', saved_pos)
            return tconstr(dshapes)
        else:
            self.advance_tok()
            ret_dshape = self.parse_datashape()
            if ret_dshape is None:
                self.raise_error('Expected function prototype return ' + 'datashape')
            tconstr = self.syntactic_sugar(self.sym.dtype_constr, 'funcproto', '(...) -> ... dtype constructor', saved_pos)
            return tconstr(dshapes, ret_dshape)