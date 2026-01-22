import re
from pygments.lexer import RegexLexer, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class CoqLexer(RegexLexer):
    """
    For the `Coq <http://coq.inria.fr/>`_ theorem prover.

    .. versionadded:: 1.5
    """
    name = 'Coq'
    aliases = ['coq']
    filenames = ['*.v']
    mimetypes = ['text/x-coq']
    keywords1 = ('Section', 'Module', 'End', 'Require', 'Import', 'Export', 'Variable', 'Variables', 'Parameter', 'Parameters', 'Axiom', 'Hypothesis', 'Hypotheses', 'Notation', 'Local', 'Tactic', 'Reserved', 'Scope', 'Open', 'Close', 'Bind', 'Delimit', 'Definition', 'Let', 'Ltac', 'Fixpoint', 'CoFixpoint', 'Morphism', 'Relation', 'Implicit', 'Arguments', 'Set', 'Unset', 'Contextual', 'Strict', 'Prenex', 'Implicits', 'Inductive', 'CoInductive', 'Record', 'Structure', 'Canonical', 'Coercion', 'Theorem', 'Lemma', 'Corollary', 'Proposition', 'Fact', 'Remark', 'Example', 'Proof', 'Goal', 'Save', 'Qed', 'Defined', 'Hint', 'Resolve', 'Rewrite', 'View', 'Search', 'Show', 'Print', 'Printing', 'All', 'Graph', 'Projections', 'inside', 'outside', 'Check', 'Global', 'Instance', 'Class', 'Existing', 'Universe', 'Polymorphic', 'Monomorphic', 'Context')
    keywords2 = ('forall', 'exists', 'exists2', 'fun', 'fix', 'cofix', 'struct', 'match', 'end', 'in', 'return', 'let', 'if', 'is', 'then', 'else', 'for', 'of', 'nosimpl', 'with', 'as')
    keywords3 = ('Type', 'Prop')
    keywords4 = ('pose', 'set', 'move', 'case', 'elim', 'apply', 'clear', 'hnf', 'intro', 'intros', 'generalize', 'rename', 'pattern', 'after', 'destruct', 'induction', 'using', 'refine', 'inversion', 'injection', 'rewrite', 'congr', 'unlock', 'compute', 'ring', 'field', 'replace', 'fold', 'unfold', 'change', 'cutrewrite', 'simpl', 'have', 'suff', 'wlog', 'suffices', 'without', 'loss', 'nat_norm', 'assert', 'cut', 'trivial', 'revert', 'bool_congr', 'nat_congr', 'symmetry', 'transitivity', 'auto', 'split', 'left', 'right', 'autorewrite', 'tauto', 'setoid_rewrite', 'intuition', 'eauto', 'eapply', 'econstructor', 'etransitivity', 'constructor', 'erewrite', 'red', 'cbv', 'lazy', 'vm_compute', 'native_compute', 'subst')
    keywords5 = ('by', 'done', 'exact', 'reflexivity', 'tauto', 'romega', 'omega', 'assumption', 'solve', 'contradiction', 'discriminate', 'congruence')
    keywords6 = ('do', 'last', 'first', 'try', 'idtac', 'repeat')
    keyopts = ('!=', '#', '&', '&&', '\\(', '\\)', '\\*', '\\+', ',', '-', '-\\.', '->', '\\.', '\\.\\.', ':', '::', ':=', ':>', ';', ';;', '<', '<-', '<->', '=', '>', '>]', '>\\}', '\\?', '\\?\\?', '\\[', '\\[<', '\\[>', '\\[\\|', ']', '_', '`', '\\{', '\\{<', '\\|', '\\|]', '\\}', '~', '=>', '/\\\\', '\\\\/', '\\{\\|', '\\|\\}', u'Π', u'λ')
    operators = '[!$%&*+\\./:<=>?@^|~-]'
    prefix_syms = '[!?~]'
    infix_syms = '[=<>@^|&+\\*/$%-]'
    primitives = ('unit', 'nat', 'bool', 'string', 'ascii', 'list')
    tokens = {'root': [('\\s+', Text), ('false|true|\\(\\)|\\[\\]', Name.Builtin.Pseudo), ('\\(\\*', Comment, 'comment'), (words(keywords1, prefix='\\b', suffix='\\b'), Keyword.Namespace), (words(keywords2, prefix='\\b', suffix='\\b'), Keyword), (words(keywords3, prefix='\\b', suffix='\\b'), Keyword.Type), (words(keywords4, prefix='\\b', suffix='\\b'), Keyword), (words(keywords5, prefix='\\b', suffix='\\b'), Keyword.Pseudo), (words(keywords6, prefix='\\b', suffix='\\b'), Keyword.Reserved), ("\\b([A-Z][\\w\\']*)", Name), ('(%s)' % '|'.join(keyopts[::-1]), Operator), ('(%s|%s)?%s' % (infix_syms, prefix_syms, operators), Operator), ('\\b(%s)\\b' % '|'.join(primitives), Keyword.Type), ("[^\\W\\d][\\w']*", Name), ('\\d[\\d_]*', Number.Integer), ('0[xX][\\da-fA-F][\\da-fA-F_]*', Number.Hex), ('0[oO][0-7][0-7_]*', Number.Oct), ('0[bB][01][01_]*', Number.Bin), ('-?\\d[\\d_]*(.[\\d_]*)?([eE][+\\-]?\\d[\\d_]*)', Number.Float), ('\'(?:(\\\\[\\\\\\"\'ntbr ])|(\\\\[0-9]{3})|(\\\\x[0-9a-fA-F]{2}))\'', String.Char), ("'.'", String.Char), ("'", Keyword), ('"', String.Double, 'string'), ("[~?][a-z][\\w\\']*:", Name)], 'comment': [('[^(*)]+', Comment), ('\\(\\*', Comment, '#push'), ('\\*\\)', Comment, '#pop'), ('[(*)]', Comment)], 'string': [('[^"]+', String.Double), ('""', String.Double), ('"', String.Double, '#pop')], 'dotted': [('\\s+', Text), ('\\.', Punctuation), ("[A-Z][\\w\\']*(?=\\s*\\.)", Name.Namespace), ("[A-Z][\\w\\']*", Name.Class, '#pop'), ("[a-z][a-z0-9_\\']*", Name, '#pop'), default('#pop')]}

    def analyse_text(text):
        if text.startswith('(*'):
            return True