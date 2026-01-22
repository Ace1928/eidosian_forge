import functools
import operator
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Union, TYPE_CHECKING
import numpy as np
import sympy
from ply import yacc
from cirq import ops, Circuit, NamedQubit, CX
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import._lexer import QasmLexer
from cirq.contrib.qasm_import.exception import QasmException
class QasmParser:
    """Parser for QASM strings.

    Example:

        qasm = "OPENQASM 2.0; qreg q1[2]; CX q1[0], q1[1];"
        parsedQasm = QasmParser().parse(qasm)
    """

    def __init__(self) -> None:
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)
        self.circuit = Circuit()
        self.qregs: Dict[str, int] = {}
        self.cregs: Dict[str, int] = {}
        self.qelibinc = False
        self.lexer = QasmLexer()
        self.supported_format = False
        self.parsedQasm: Optional[Qasm] = None
        self.qubits: Dict[str, ops.Qid] = {}
        self.functions = {'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'exp': np.exp, 'ln': np.log, 'sqrt': np.sqrt, 'acos': np.arccos, 'atan': np.arctan, 'asin': np.arcsin}
        self.binary_operators = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv, '^': operator.pow}
    basic_gates: Dict[str, QasmGateStatement] = {'CX': QasmGateStatement(qasm_gate='CX', cirq_gate=CX, num_params=0, num_args=2), 'U': QasmGateStatement(qasm_gate='U', num_params=3, num_args=1, cirq_gate=lambda params: QasmUGate(*[p / np.pi for p in params]))}
    qelib_gates = {'rx': QasmGateStatement(qasm_gate='rx', cirq_gate=lambda params: ops.rx(params[0]), num_params=1, num_args=1), 'sx': QasmGateStatement(qasm_gate='sx', num_params=0, num_args=1, cirq_gate=ops.XPowGate(exponent=0.5)), 'sxdg': QasmGateStatement(qasm_gate='sxdg', num_params=0, num_args=1, cirq_gate=ops.XPowGate(exponent=-0.5)), 'ry': QasmGateStatement(qasm_gate='ry', cirq_gate=lambda params: ops.ry(params[0]), num_params=1, num_args=1), 'rz': QasmGateStatement(qasm_gate='rz', cirq_gate=lambda params: ops.rz(params[0]), num_params=1, num_args=1), 'id': QasmGateStatement(qasm_gate='id', cirq_gate=ops.IdentityGate(1), num_params=0, num_args=1), 'u1': QasmGateStatement(qasm_gate='u1', cirq_gate=lambda params: QasmUGate(0, 0, params[0] / np.pi), num_params=1, num_args=1), 'u2': QasmGateStatement(qasm_gate='u2', cirq_gate=lambda params: QasmUGate(0.5, params[0] / np.pi, params[1] / np.pi), num_params=2, num_args=1), 'u3': QasmGateStatement(qasm_gate='u3', num_params=3, num_args=1, cirq_gate=lambda params: QasmUGate(*[p / np.pi for p in params])), 'r': QasmGateStatement(qasm_gate='r', num_params=2, num_args=1, cirq_gate=lambda params: QasmUGate(params[0] / np.pi, params[1] / np.pi - 0.5, -params[1] / np.pi + 0.5)), 'x': QasmGateStatement(qasm_gate='x', num_params=0, num_args=1, cirq_gate=ops.X), 'y': QasmGateStatement(qasm_gate='y', num_params=0, num_args=1, cirq_gate=ops.Y), 'z': QasmGateStatement(qasm_gate='z', num_params=0, num_args=1, cirq_gate=ops.Z), 'h': QasmGateStatement(qasm_gate='h', num_params=0, num_args=1, cirq_gate=ops.H), 's': QasmGateStatement(qasm_gate='s', num_params=0, num_args=1, cirq_gate=ops.S), 't': QasmGateStatement(qasm_gate='t', num_params=0, num_args=1, cirq_gate=ops.T), 'cx': QasmGateStatement(qasm_gate='cx', cirq_gate=CX, num_params=0, num_args=2), 'cy': QasmGateStatement(qasm_gate='cy', cirq_gate=ops.ControlledGate(ops.Y), num_params=0, num_args=2), 'cz': QasmGateStatement(qasm_gate='cz', cirq_gate=ops.CZ, num_params=0, num_args=2), 'ch': QasmGateStatement(qasm_gate='ch', cirq_gate=ops.ControlledGate(ops.H), num_params=0, num_args=2), 'swap': QasmGateStatement(qasm_gate='swap', cirq_gate=ops.SWAP, num_params=0, num_args=2), 'cswap': QasmGateStatement(qasm_gate='cswap', num_params=0, num_args=3, cirq_gate=ops.CSWAP), 'ccx': QasmGateStatement(qasm_gate='ccx', num_params=0, num_args=3, cirq_gate=ops.CCX), 'sdg': QasmGateStatement(qasm_gate='sdg', num_params=0, num_args=1, cirq_gate=ops.S ** (-1)), 'tdg': QasmGateStatement(qasm_gate='tdg', num_params=0, num_args=1, cirq_gate=ops.T ** (-1))}
    all_gates = {**basic_gates, **qelib_gates}
    tokens = QasmLexer.tokens
    start = 'start'
    precedence = (('left', '+', '-'), ('left', '*', '/'), ('right', '^'))

    def p_start(self, p):
        """start : qasm"""
        p[0] = p[1]

    def p_qasm_format_only(self, p):
        """qasm : format"""
        self.supported_format = True
        p[0] = Qasm(self.supported_format, self.qelibinc, self.qregs, self.cregs, self.circuit)

    def p_qasm_no_format_specified_error(self, p):
        """qasm : QELIBINC
        | circuit"""
        if self.supported_format is False:
            raise QasmException("Missing 'OPENQASM 2.0;' statement")

    def p_qasm_include(self, p):
        """qasm : qasm QELIBINC"""
        self.qelibinc = True
        p[0] = Qasm(self.supported_format, self.qelibinc, self.qregs, self.cregs, self.circuit)

    def p_qasm_circuit(self, p):
        """qasm : qasm circuit"""
        p[0] = Qasm(self.supported_format, self.qelibinc, self.qregs, self.cregs, p[2])

    def p_format(self, p):
        """format : FORMAT_SPEC"""
        if p[1] != '2.0':
            raise QasmException(f'Unsupported OpenQASM version: {p[1]}, only 2.0 is supported currently by Cirq')

    def p_circuit_reg(self, p):
        """circuit : new_reg circuit"""
        p[0] = self.circuit

    def p_circuit_gate_or_measurement_or_if(self, p):
        """circuit :  circuit gate_op
        |  circuit measurement
        |  circuit if"""
        self.circuit.append(p[2])
        p[0] = self.circuit

    def p_circuit_empty(self, p):
        """circuit : empty"""
        p[0] = self.circuit

    def p_new_reg(self, p):
        """new_reg : QREG ID '[' NATURAL_NUMBER ']' ';'
        | CREG ID '[' NATURAL_NUMBER ']' ';'"""
        name, length = (p[2], p[4])
        if name in self.qregs.keys() or name in self.cregs.keys():
            raise QasmException(f'{name} is already defined at line {p.lineno(2)}')
        if length == 0:
            raise QasmException(f"Illegal, zero-length register '{name}' at line {p.lineno(4)}")
        if p[1] == 'qreg':
            self.qregs[name] = length
        else:
            self.cregs[name] = length
        p[0] = (name, length)

    def p_gate_op_no_params(self, p):
        """gate_op :  ID qargs"""
        self._resolve_gate_operation(p[2], gate=p[1], p=p, params=[])

    def p_gate_op_with_params(self, p):
        """gate_op :  ID '(' params ')' qargs"""
        self._resolve_gate_operation(args=p[5], gate=p[1], p=p, params=p[3])

    def _resolve_gate_operation(self, args: List[List[ops.Qid]], gate: str, p: Any, params: List[float]):
        gate_set = self.basic_gates if not self.qelibinc else self.all_gates
        if gate not in gate_set.keys():
            tip = ', did you forget to include qelib1.inc?' if not self.qelibinc else ''
            msg = f'Unknown gate "{gate}" at line {p.lineno(1)}{tip}'
            raise QasmException(msg)
        p[0] = gate_set[gate].on(args=args, params=params, lineno=p.lineno(1))

    def p_params_multiple(self, p):
        """params : expr ',' params"""
        p[3].insert(0, p[1])
        p[0] = p[3]

    def p_params_single(self, p):
        """params : expr"""
        p[0] = [p[1]]

    def p_expr_term(self, p):
        """expr : term"""
        p[0] = p[1]

    def p_expr_parens(self, p):
        """expr : '(' expr ')'"""
        p[0] = p[2]

    def p_expr_function_call(self, p):
        """expr : ID '(' expr ')'"""
        func = p[1]
        if func not in self.functions.keys():
            raise QasmException(f"Function not recognized: '{func}' at line {p.lineno(1)}")
        p[0] = self.functions[func](p[3])

    def p_expr_unary(self, p):
        """expr : '-' expr
        | '+' expr"""
        if p[1] == '-':
            p[0] = -p[2]
        else:
            p[0] = p[2]

    def p_expr_binary(self, p):
        """expr : expr '*' expr
        | expr '/' expr
        | expr '+' expr
        | expr '-' expr
        | expr '^' expr
        """
        p[0] = self.binary_operators[p[2]](p[1], p[3])

    def p_term(self, p):
        """term : NUMBER
        | NATURAL_NUMBER
        | PI"""
        p[0] = p[1]

    def p_args_multiple(self, p):
        """qargs : qarg ',' qargs"""
        p[3].insert(0, p[1])
        p[0] = p[3]

    def p_args_single(self, p):
        """qargs : qarg ';'"""
        p[0] = [p[1]]

    def p_quantum_arg_register(self, p):
        """qarg : ID"""
        reg = p[1]
        if reg not in self.qregs.keys():
            raise QasmException(f'Undefined quantum register "{reg}" at line {p.lineno(1)}')
        qubits = []
        for idx in range(self.qregs[reg]):
            arg_name = self.make_name(idx, reg)
            if arg_name not in self.qubits.keys():
                self.qubits[arg_name] = NamedQubit(arg_name)
            qubits.append(self.qubits[arg_name])
        p[0] = qubits

    def p_classical_arg_register(self, p):
        """carg : ID"""
        reg = p[1]
        if reg not in self.cregs.keys():
            raise QasmException(f'Undefined classical register "{reg}" at line {p.lineno(1)}')
        p[0] = [self.make_name(idx, reg) for idx in range(self.cregs[reg])]

    def make_name(self, idx, reg):
        return str(reg) + '_' + str(idx)

    def p_quantum_arg_bit(self, p):
        """qarg : ID '[' NATURAL_NUMBER ']'"""
        reg = p[1]
        idx = p[3]
        arg_name = self.make_name(idx, reg)
        if reg not in self.qregs.keys():
            raise QasmException(f'Undefined quantum register "{reg}" at line {p.lineno(1)}')
        size = self.qregs[reg]
        if idx >= size:
            raise QasmException(f'Out of bounds qubit index {idx} on register {reg} of size {size} at line {p.lineno(1)}')
        if arg_name not in self.qubits.keys():
            self.qubits[arg_name] = NamedQubit(arg_name)
        p[0] = [self.qubits[arg_name]]

    def p_classical_arg_bit(self, p):
        """carg : ID '[' NATURAL_NUMBER ']'"""
        reg = p[1]
        idx = p[3]
        arg_name = self.make_name(idx, reg)
        if reg not in self.cregs.keys():
            raise QasmException(f'Undefined classical register "{reg}" at line {p.lineno(1)}')
        size = self.cregs[reg]
        if idx >= size:
            raise QasmException(f'Out of bounds bit index {idx} on classical register {reg} of size {size} at line {p.lineno(1)}')
        p[0] = [arg_name]

    def p_measurement(self, p):
        """measurement : MEASURE qarg ARROW carg ';'"""
        qreg = p[2]
        creg = p[4]
        if len(qreg) != len(creg):
            raise QasmException(f'mismatched register sizes {len(qreg)} -> {len(creg)} for measurement at line {p.lineno(1)}')
        p[0] = [ops.MeasurementGate(num_qubits=1, key=creg[i]).on(qreg[i]) for i in range(len(qreg))]

    def p_if(self, p):
        """if : IF '(' carg EQ NATURAL_NUMBER ')' gate_op"""
        conditions = []
        for i, key in enumerate(p[3]):
            v = p[5] >> i & 1
            conditions.append(sympy.Eq(sympy.Symbol(key), v))
        p[0] = [ops.ClassicallyControlledOperation(conditions=conditions, sub_operation=tuple(p[7])[0])]

    def p_error(self, p):
        if p is None:
            raise QasmException('Unexpected end of file')
        raise QasmException(f"Syntax error: '{p.value}'\n{self.debug_context(p)}\nat line {p.lineno}, column {self.find_column(p)}")

    def find_column(self, p):
        line_start = self.qasm.rfind('\n', 0, p.lexpos) + 1
        return p.lexpos - line_start + 1

    def p_empty(self, p):
        """empty :"""

    def parse(self, qasm: str) -> Qasm:
        if self.parsedQasm is None:
            self.qasm = qasm
            self.lexer.input(self.qasm)
            self.parsedQasm = self.parser.parse(lexer=self.lexer)
        return self.parsedQasm

    def debug_context(self, p):
        debug_start = max(self.qasm.rfind('\n', 0, p.lexpos) + 1, p.lexpos - 5)
        debug_end = min(self.qasm.find('\n', p.lexpos, p.lexpos + 5), p.lexpos + 5)
        return '...' + self.qasm[debug_start:debug_end] + '\n' + ' ' * (3 + p.lexpos - debug_start) + '^'