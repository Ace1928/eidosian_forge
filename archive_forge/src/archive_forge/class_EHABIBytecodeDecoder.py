from collections import namedtuple
class EHABIBytecodeDecoder(object):
    """ Decoder of a sequence of ARM exception handler abi bytecode.

        Reference:
        https://github.com/llvm/llvm-project/blob/master/llvm/tools/llvm-readobj/ARMEHABIPrinter.h
        https://developer.arm.com/documentation/ihi0038/b/

        Accessible attributes:

            mnemonic_array:
                MnemonicItem array.

        Parameters:

            bytecode_array:
                Integer array, raw data of bytecode.

    """

    def __init__(self, bytecode_array):
        self._bytecode_array = bytecode_array
        self._index = None
        self.mnemonic_array = None
        self._decode()

    def _decode(self):
        """ Decode bytecode array, put result into mnemonic_array.
        """
        self._index = 0
        self.mnemonic_array = []
        while self._index < len(self._bytecode_array):
            for mask, value, handler in self.ring:
                if self._bytecode_array[self._index] & mask == value:
                    start_idx = self._index
                    mnemonic = handler(self)
                    end_idx = self._index
                    self.mnemonic_array.append(MnemonicItem(self._bytecode_array[start_idx:end_idx], mnemonic))
                    break

    def _decode_00xxxxxx(self):
        opcode = self._bytecode_array[self._index]
        self._index += 1
        return 'vsp = vsp + %u' % (((opcode & 63) << 2) + 4)

    def _decode_01xxxxxx(self):
        opcode = self._bytecode_array[self._index]
        self._index += 1
        return 'vsp = vsp - %u' % (((opcode & 63) << 2) + 4)
    gpr_register_names = ('r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'fp', 'ip', 'sp', 'lr', 'pc')

    def _calculate_range(self, start, count):
        return (1 << count + 1) - 1 << start

    def _printGPR(self, gpr_mask):
        hits = [self.gpr_register_names[i] for i in range(32) if gpr_mask & 1 << i != 0]
        return '{%s}' % ', '.join(hits)

    def _print_registers(self, vfp_mask, prefix):
        hits = [prefix + str(i) for i in range(32) if vfp_mask & 1 << i != 0]
        return '{%s}' % ', '.join(hits)

    def _decode_1000iiii_iiiiiiii(self):
        op0 = self._bytecode_array[self._index]
        self._index += 1
        op1 = self._bytecode_array[self._index]
        self._index += 1
        gpr_mask = op1 << 4 | (op0 & 15) << 12
        if gpr_mask == 0:
            return 'refuse to unwind'
        else:
            return 'pop %s' % self._printGPR(gpr_mask)

    def _decode_10011101(self):
        self._index += 1
        return 'reserved (ARM MOVrr)'

    def _decode_10011111(self):
        self._index += 1
        return 'reserved (WiMMX MOVrr)'

    def _decode_1001nnnn(self):
        opcode = self._bytecode_array[self._index]
        self._index += 1
        return 'vsp = r%u' % (opcode & 15)

    def _decode_10100nnn(self):
        opcode = self._bytecode_array[self._index]
        self._index += 1
        return 'pop %s' % self._printGPR(self._calculate_range(4, opcode & 7))

    def _decode_10101nnn(self):
        opcode = self._bytecode_array[self._index]
        self._index += 1
        return 'pop %s' % self._printGPR(self._calculate_range(4, opcode & 7) | 1 << 14)

    def _decode_10110000(self):
        self._index += 1
        return 'finish'

    def _decode_10110001_0000iiii(self):
        self._index += 1
        op1 = self._bytecode_array[self._index]
        self._index += 1
        if op1 & 240 != 0 or op1 == 0:
            return 'spare'
        else:
            return 'pop %s' % self._printGPR(op1 & 15)

    def _decode_10110010_uleb128(self):
        self._index += 1
        uleb_buffer = [self._bytecode_array[self._index]]
        self._index += 1
        while self._bytecode_array[self._index] & 128 == 0:
            uleb_buffer.append(self._bytecode_array[self._index])
            self._index += 1
        value = 0
        for b in reversed(uleb_buffer):
            value = (value << 7) + (b & 127)
        return 'vsp = vsp + %u' % (516 + (value << 2))

    def _decode_10110011_sssscccc(self):
        return self._decode_11001001_sssscccc()

    def _decode_101101nn(self):
        return self._spare()

    def _decode_10111nnn(self):
        opcode = self._bytecode_array[self._index]
        self._index += 1
        return 'pop %s' % self._print_registers(self._calculate_range(8, opcode & 7), 'd')

    def _decode_11000110_sssscccc(self):
        self._index += 1
        op1 = self._bytecode_array[self._index]
        self._index += 1
        start = (op1 & 240) >> 4
        count = (op1 & 15) >> 0
        return 'pop %s' % self._print_registers(self._calculate_range(start, count), 'wR')

    def _decode_11000111_0000iiii(self):
        self._index += 1
        op1 = self._bytecode_array[self._index]
        self._index += 1
        if op1 & 240 != 0 or op1 == 0:
            return 'spare'
        else:
            return 'pop %s' % self._print_registers(op1 & 15, 'wCGR')

    def _decode_11001000_sssscccc(self):
        self._index += 1
        op1 = self._bytecode_array[self._index]
        self._index += 1
        start = 16 + ((op1 & 240) >> 4)
        count = (op1 & 15) >> 0
        return 'pop %s' % self._print_registers(self._calculate_range(start, count), 'd')

    def _decode_11001001_sssscccc(self):
        self._index += 1
        op1 = self._bytecode_array[self._index]
        self._index += 1
        start = (op1 & 240) >> 4
        count = (op1 & 15) >> 0
        return 'pop %s' % self._print_registers(self._calculate_range(start, count), 'd')

    def _decode_11001yyy(self):
        return self._spare()

    def _decode_11000nnn(self):
        opcode = self._bytecode_array[self._index]
        self._index += 1
        return 'pop %s' % self._print_registers(self._calculate_range(10, opcode & 7), 'wR')

    def _decode_11010nnn(self):
        return self._decode_10111nnn()

    def _decode_11xxxyyy(self):
        return self._spare()

    def _spare(self):
        self._index += 1
        return 'spare'
    _DECODE_RECIPE_TYPE = namedtuple('_DECODE_RECIPE_TYPE', 'mask value handler')
    ring = (_DECODE_RECIPE_TYPE(mask=192, value=0, handler=_decode_00xxxxxx), _DECODE_RECIPE_TYPE(mask=192, value=64, handler=_decode_01xxxxxx), _DECODE_RECIPE_TYPE(mask=240, value=128, handler=_decode_1000iiii_iiiiiiii), _DECODE_RECIPE_TYPE(mask=255, value=157, handler=_decode_10011101), _DECODE_RECIPE_TYPE(mask=255, value=159, handler=_decode_10011111), _DECODE_RECIPE_TYPE(mask=240, value=144, handler=_decode_1001nnnn), _DECODE_RECIPE_TYPE(mask=248, value=160, handler=_decode_10100nnn), _DECODE_RECIPE_TYPE(mask=248, value=168, handler=_decode_10101nnn), _DECODE_RECIPE_TYPE(mask=255, value=176, handler=_decode_10110000), _DECODE_RECIPE_TYPE(mask=255, value=177, handler=_decode_10110001_0000iiii), _DECODE_RECIPE_TYPE(mask=255, value=178, handler=_decode_10110010_uleb128), _DECODE_RECIPE_TYPE(mask=255, value=179, handler=_decode_10110011_sssscccc), _DECODE_RECIPE_TYPE(mask=252, value=180, handler=_decode_101101nn), _DECODE_RECIPE_TYPE(mask=248, value=184, handler=_decode_10111nnn), _DECODE_RECIPE_TYPE(mask=255, value=198, handler=_decode_11000110_sssscccc), _DECODE_RECIPE_TYPE(mask=255, value=199, handler=_decode_11000111_0000iiii), _DECODE_RECIPE_TYPE(mask=255, value=200, handler=_decode_11001000_sssscccc), _DECODE_RECIPE_TYPE(mask=255, value=201, handler=_decode_11001001_sssscccc), _DECODE_RECIPE_TYPE(mask=200, value=200, handler=_decode_11001yyy), _DECODE_RECIPE_TYPE(mask=248, value=192, handler=_decode_11000nnn), _DECODE_RECIPE_TYPE(mask=248, value=208, handler=_decode_11010nnn), _DECODE_RECIPE_TYPE(mask=192, value=192, handler=_decode_11xxxyyy))