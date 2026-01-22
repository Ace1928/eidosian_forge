from typing import Optional
class InetdService:
    """
    A simple description of an inetd service.
    """
    name = None
    port = None
    socketType = None
    protocol = None
    wait = None
    user = None
    group = None
    program = None
    programArgs = None

    def __init__(self, name, port, socketType, protocol, wait, user, group, program, programArgs):
        self.name = name
        self.port = port
        self.socketType = socketType
        self.protocol = protocol
        self.wait = wait
        self.user = user
        self.group = group
        self.program = program
        self.programArgs = programArgs