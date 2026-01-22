import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def _var_type(self, lltype, size, datamodel=None):
    if self._DEBUG:
        print('-->', lltype, size, datamodel, getattr(datamodel, 'fe_type', 'NO FE TYPE'))
    m = self.module
    bitsize = _BYTE_SIZE * size
    int_type = (ir.IntType,)
    real_type = (ir.FloatType, ir.DoubleType)
    if isinstance(lltype, int_type + real_type):
        if datamodel is None:
            name = str(lltype)
            if isinstance(lltype, int_type):
                ditok = 'DW_ATE_unsigned'
            else:
                ditok = 'DW_ATE_float'
        else:
            name = str(datamodel.fe_type)
            if isinstance(datamodel.fe_type, types.Integer):
                if datamodel.fe_type.signed:
                    ditok = 'DW_ATE_signed'
                else:
                    ditok = 'DW_ATE_unsigned'
            else:
                ditok = 'DW_ATE_float'
        mdtype = m.add_debug_info('DIBasicType', {'name': name, 'size': bitsize, 'encoding': ir.DIToken(ditok)})
    elif isinstance(datamodel, ComplexModel):
        meta = []
        offset = 0
        for ix, name in enumerate(('real', 'imag')):
            component = lltype.elements[ix]
            component_size = self.cgctx.get_abi_sizeof(component)
            component_basetype = m.add_debug_info('DIBasicType', {'name': str(component), 'size': _BYTE_SIZE * component_size, 'encoding': ir.DIToken('DW_ATE_float')})
            derived_type = m.add_debug_info('DIDerivedType', {'tag': ir.DIToken('DW_TAG_member'), 'name': name, 'baseType': component_basetype, 'size': _BYTE_SIZE * component_size, 'offset': offset})
            meta.append(derived_type)
            offset += _BYTE_SIZE * component_size
        mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_structure_type'), 'name': f'{datamodel.fe_type} ({str(lltype)})', 'identifier': str(lltype), 'elements': m.add_metadata(meta), 'size': offset}, is_distinct=True)
    elif isinstance(datamodel, UniTupleModel):
        element = lltype.element
        el_size = self.cgctx.get_abi_sizeof(element)
        basetype = self._var_type(element, el_size)
        name = f'{datamodel.fe_type} ({str(lltype)})'
        count = size // el_size
        mdrange = m.add_debug_info('DISubrange', {'count': count})
        mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_array_type'), 'baseType': basetype, 'name': name, 'size': bitsize, 'identifier': str(lltype), 'elements': m.add_metadata([mdrange])})
    elif isinstance(lltype, ir.PointerType):
        model = getattr(datamodel, '_pointee_model', None)
        basetype = self._var_type(lltype.pointee, self.cgctx.get_abi_sizeof(lltype.pointee), model)
        mdtype = m.add_debug_info('DIDerivedType', {'tag': ir.DIToken('DW_TAG_pointer_type'), 'baseType': basetype, 'size': _BYTE_SIZE * self.cgctx.get_abi_sizeof(lltype)})
    elif isinstance(lltype, ir.LiteralStructType):
        meta = []
        offset = 0
        if datamodel is None or not datamodel.inner_models():
            name = f'Anonymous struct ({str(lltype)})'
            for field_id, element in enumerate(lltype.elements):
                size = self.cgctx.get_abi_sizeof(element)
                basetype = self._var_type(element, size)
                derived_type = m.add_debug_info('DIDerivedType', {'tag': ir.DIToken('DW_TAG_member'), 'name': f'<field {field_id}>', 'baseType': basetype, 'size': _BYTE_SIZE * size, 'offset': offset})
                meta.append(derived_type)
                offset += _BYTE_SIZE * size
        else:
            name = f'{datamodel.fe_type} ({str(lltype)})'
            for element, field, model in zip(lltype.elements, datamodel._fields, datamodel.inner_models()):
                size = self.cgctx.get_abi_sizeof(element)
                basetype = self._var_type(element, size, datamodel=model)
                derived_type = m.add_debug_info('DIDerivedType', {'tag': ir.DIToken('DW_TAG_member'), 'name': field, 'baseType': basetype, 'size': _BYTE_SIZE * size, 'offset': offset})
                meta.append(derived_type)
                offset += _BYTE_SIZE * size
        mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_structure_type'), 'name': name, 'identifier': str(lltype), 'elements': m.add_metadata(meta), 'size': offset}, is_distinct=True)
    elif isinstance(lltype, ir.ArrayType):
        element = lltype.element
        el_size = self.cgctx.get_abi_sizeof(element)
        basetype = self._var_type(element, el_size)
        count = size // el_size
        mdrange = m.add_debug_info('DISubrange', {'count': count})
        mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_array_type'), 'baseType': basetype, 'name': str(lltype), 'size': bitsize, 'identifier': str(lltype), 'elements': m.add_metadata([mdrange])})
    else:
        count = size
        mdrange = m.add_debug_info('DISubrange', {'count': count})
        mdbase = m.add_debug_info('DIBasicType', {'name': 'byte', 'size': _BYTE_SIZE, 'encoding': ir.DIToken('DW_ATE_unsigned_char')})
        mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_array_type'), 'baseType': mdbase, 'name': str(lltype), 'size': bitsize, 'identifier': str(lltype), 'elements': m.add_metadata([mdrange])})
    return mdtype