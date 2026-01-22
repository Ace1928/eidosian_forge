import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumGateCall(QuantumInstruction):
    """
    quantumGateCall
        : quantumGateModifier* quantumGateName ( LPAREN expressionList? RPAREN )? indexIdentifierList
    """

    def __init__(self, quantumGateName: Identifier, indexIdentifierList: List[Identifier], parameters: List[Expression]=None, modifiers: Optional[List[QuantumGateModifier]]=None):
        self.quantumGateName = quantumGateName
        self.indexIdentifierList = indexIdentifierList
        self.parameters = parameters or []
        self.modifiers = modifiers or []