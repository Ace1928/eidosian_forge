from ..construct import UBInt32, ULInt32, Struct
def _create_exception_handler_table(self):
    self.EH_table_struct = Struct('EH_table', self.EHABI_uint32('word0'))