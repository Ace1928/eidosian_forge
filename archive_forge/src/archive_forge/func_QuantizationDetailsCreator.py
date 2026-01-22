import flatbuffers
from flatbuffers.compat import import_numpy
def QuantizationDetailsCreator(unionType, table):
    from flatbuffers.table import Table
    if not isinstance(table, Table):
        return None
    if unionType == QuantizationDetails().CustomQuantization:
        return CustomQuantizationT.InitFromBuf(table.Bytes, table.Pos)
    return None