import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumGateSignature(ASTNode):
    """
    quantumGateSignature
        : quantumGateName ( LPAREN identifierList? RPAREN )? identifierList
    """

    def __init__(self, name: Identifier, qargList: List[Identifier], params: Optional[List[Expression]]=None):
        self.name = name
        self.qargList = qargList
        self.params = params