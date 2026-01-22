from pygments.lexer import RegexLexer, include, bygroups
from pygments.token import Generic, Comment, String, Text, Keyword, Name, \
class PyPyLogLexer(RegexLexer):
    """
    Lexer for PyPy log files.

    .. versionadded:: 1.5
    """
    name = 'PyPy Log'
    aliases = ['pypylog', 'pypy']
    filenames = ['*.pypylog']
    mimetypes = ['application/x-pypylog']
    tokens = {'root': [('\\[\\w+\\] \\{jit-log-.*?$', Keyword, 'jit-log'), ('\\[\\w+\\] \\{jit-backend-counts$', Keyword, 'jit-backend-counts'), include('extra-stuff')], 'jit-log': [('\\[\\w+\\] jit-log-.*?}$', Keyword, '#pop'), ('^\\+\\d+: ', Comment), ('--end of the loop--', Comment), ('[ifp]\\d+', Name), ('ptr\\d+', Name), ('(\\()(\\w+(?:\\.\\w+)?)(\\))', bygroups(Punctuation, Name.Builtin, Punctuation)), ('[\\[\\]=,()]', Punctuation), ('(\\d+\\.\\d+|inf|-inf)', Number.Float), ('-?\\d+', Number.Integer), ("'.*'", String), ('(None|descr|ConstClass|ConstPtr|TargetToken)', Name), ('<.*?>+', Name.Builtin), ('(label|debug_merge_point|jump|finish)', Name.Class), ('(int_add_ovf|int_add|int_sub_ovf|int_sub|int_mul_ovf|int_mul|int_floordiv|int_mod|int_lshift|int_rshift|int_and|int_or|int_xor|int_eq|int_ne|int_ge|int_gt|int_le|int_lt|int_is_zero|int_is_true|uint_floordiv|uint_ge|uint_lt|float_add|float_sub|float_mul|float_truediv|float_neg|float_eq|float_ne|float_ge|float_gt|float_le|float_lt|float_abs|ptr_eq|ptr_ne|instance_ptr_eq|instance_ptr_ne|cast_int_to_float|cast_float_to_int|force_token|quasiimmut_field|same_as|virtual_ref_finish|virtual_ref|mark_opaque_ptr|call_may_force|call_assembler|call_loopinvariant|call_release_gil|call_pure|call|new_with_vtable|new_array|newstr|newunicode|new|arraylen_gc|getarrayitem_gc_pure|getarrayitem_gc|setarrayitem_gc|getarrayitem_raw|setarrayitem_raw|getfield_gc_pure|getfield_gc|getinteriorfield_gc|setinteriorfield_gc|getfield_raw|setfield_gc|setfield_raw|strgetitem|strsetitem|strlen|copystrcontent|unicodegetitem|unicodesetitem|unicodelen|guard_true|guard_false|guard_value|guard_isnull|guard_nonnull_class|guard_nonnull|guard_class|guard_no_overflow|guard_not_forced|guard_no_exception|guard_not_invalidated)', Name.Builtin), include('extra-stuff')], 'jit-backend-counts': [('\\[\\w+\\] jit-backend-counts}$', Keyword, '#pop'), (':', Punctuation), ('\\d+', Number), include('extra-stuff')], 'extra-stuff': [('\\s+', Text), ('#.*?$', Comment)]}