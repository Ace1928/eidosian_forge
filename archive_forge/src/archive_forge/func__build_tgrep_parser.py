import functools
import re
import nltk.tree
def _build_tgrep_parser(set_parse_actions=True):
    """
    Builds a pyparsing-based parser object for tokenizing and
    interpreting tgrep search strings.
    """
    tgrep_op = pyparsing.Optional('!') + pyparsing.Regex("[$%,.<>][%,.<>0-9-':]*")
    tgrep_qstring = pyparsing.QuotedString(quoteChar='"', escChar='\\', unquoteResults=False)
    tgrep_node_regex = pyparsing.QuotedString(quoteChar='/', escChar='\\', unquoteResults=False)
    tgrep_qstring_icase = pyparsing.Regex('i@\\"(?:[^"\\n\\r\\\\]|(?:\\\\.))*\\"')
    tgrep_node_regex_icase = pyparsing.Regex('i@\\/(?:[^/\\n\\r\\\\]|(?:\\\\.))*\\/')
    tgrep_node_literal = pyparsing.Regex("[^][ \r\t\n;:.,&|<>()$!@%'^=]+")
    tgrep_expr = pyparsing.Forward()
    tgrep_relations = pyparsing.Forward()
    tgrep_parens = pyparsing.Literal('(') + tgrep_expr + ')'
    tgrep_nltk_tree_pos = pyparsing.Literal('N(') + pyparsing.Optional(pyparsing.Word(pyparsing.nums) + ',' + pyparsing.Optional(pyparsing.delimitedList(pyparsing.Word(pyparsing.nums), delim=',') + pyparsing.Optional(','))) + ')'
    tgrep_node_label = pyparsing.Regex('[A-Za-z0-9]+')
    tgrep_node_label_use = pyparsing.Combine('=' + tgrep_node_label)
    tgrep_node_label_use_pred = tgrep_node_label_use.copy()
    macro_name = pyparsing.Regex("[^];:.,&|<>()[$!@%'^=\r\t\n ]+")
    macro_name.setWhitespaceChars('')
    macro_use = pyparsing.Combine('@' + macro_name)
    tgrep_node_expr = tgrep_node_label_use_pred | macro_use | tgrep_nltk_tree_pos | tgrep_qstring_icase | tgrep_node_regex_icase | tgrep_qstring | tgrep_node_regex | '*' | tgrep_node_literal
    tgrep_node_expr2 = tgrep_node_expr + pyparsing.Literal('=').setWhitespaceChars('') + tgrep_node_label.copy().setWhitespaceChars('') | tgrep_node_expr
    tgrep_node = tgrep_parens | pyparsing.Optional("'") + tgrep_node_expr2 + pyparsing.ZeroOrMore('|' + tgrep_node_expr)
    tgrep_brackets = pyparsing.Optional('!') + '[' + tgrep_relations + ']'
    tgrep_relation = tgrep_brackets | tgrep_op + tgrep_node
    tgrep_rel_conjunction = pyparsing.Forward()
    tgrep_rel_conjunction << tgrep_relation + pyparsing.ZeroOrMore(pyparsing.Optional('&') + tgrep_rel_conjunction)
    tgrep_relations << tgrep_rel_conjunction + pyparsing.ZeroOrMore('|' + tgrep_relations)
    tgrep_expr << tgrep_node + pyparsing.Optional(tgrep_relations)
    tgrep_expr_labeled = tgrep_node_label_use + pyparsing.Optional(tgrep_relations)
    tgrep_expr2 = tgrep_expr + pyparsing.ZeroOrMore(':' + tgrep_expr_labeled)
    macro_defn = pyparsing.Literal('@') + pyparsing.White().suppress() + macro_name + tgrep_expr2
    tgrep_exprs = pyparsing.Optional(macro_defn + pyparsing.ZeroOrMore(';' + macro_defn) + ';') + tgrep_expr2 + pyparsing.ZeroOrMore(';' + (macro_defn | tgrep_expr2)) + pyparsing.ZeroOrMore(';').suppress()
    if set_parse_actions:
        tgrep_node_label_use.setParseAction(_tgrep_node_label_use_action)
        tgrep_node_label_use_pred.setParseAction(_tgrep_node_label_pred_use_action)
        macro_use.setParseAction(_tgrep_macro_use_action)
        tgrep_node.setParseAction(_tgrep_node_action)
        tgrep_node_expr2.setParseAction(_tgrep_bind_node_label_action)
        tgrep_parens.setParseAction(_tgrep_parens_action)
        tgrep_nltk_tree_pos.setParseAction(_tgrep_nltk_tree_pos_action)
        tgrep_relation.setParseAction(_tgrep_relation_action)
        tgrep_rel_conjunction.setParseAction(_tgrep_conjunction_action)
        tgrep_relations.setParseAction(_tgrep_rel_disjunction_action)
        macro_defn.setParseAction(_macro_defn_action)
        tgrep_expr.setParseAction(_tgrep_conjunction_action)
        tgrep_expr_labeled.setParseAction(_tgrep_segmented_pattern_action)
        tgrep_expr2.setParseAction(functools.partial(_tgrep_conjunction_action, join_char=':'))
        tgrep_exprs.setParseAction(_tgrep_exprs_action)
    return tgrep_exprs.ignore('#' + pyparsing.restOfLine)