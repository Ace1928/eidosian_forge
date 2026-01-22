import crcmod
class PredefinedCrc(crcmod.Crc):

    def __init__(self, crc_name):
        definition = _get_definition_by_name(crc_name)
        super().__init__(poly=definition['poly'], initCrc=definition['init'], rev=definition['reverse'], xorOut=definition['xor_out'])