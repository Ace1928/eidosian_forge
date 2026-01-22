class PortAlreadyExist(OSKenException):
    message = 'port (%(dpid)s, %(port)s) in network %(network_id)s already exists'