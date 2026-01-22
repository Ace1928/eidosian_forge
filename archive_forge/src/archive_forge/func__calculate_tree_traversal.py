from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def _calculate_tree_traversal(nonterminal_to_dfas):
    """
    By this point we know how dfas can move around within a stack node, but we
    don't know how we can add a new stack node (nonterminal transitions).
    """
    first_plans = {}
    nonterminals = list(nonterminal_to_dfas.keys())
    nonterminals.sort()
    for nonterminal in nonterminals:
        if nonterminal not in first_plans:
            _calculate_first_plans(nonterminal_to_dfas, first_plans, nonterminal)
    for dfas in nonterminal_to_dfas.values():
        for dfa_state in dfas:
            transitions = dfa_state.transitions
            for nonterminal, next_dfa in dfa_state.nonterminal_arcs.items():
                for transition, pushes in first_plans[nonterminal].items():
                    if transition in transitions:
                        prev_plan = transitions[transition]
                        choices = sorted([prev_plan.dfa_pushes[0].from_rule if prev_plan.dfa_pushes else prev_plan.next_dfa.from_rule, pushes[0].from_rule if pushes else next_dfa.from_rule])
                        raise ValueError("Rule %s is ambiguous; given a %s token, we can't determine if we should evaluate %s or %s." % ((dfa_state.from_rule, transition) + tuple(choices)))
                    transitions[transition] = DFAPlan(next_dfa, pushes)