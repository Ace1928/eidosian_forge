from types import SimpleNamespace
from fontTools.misc.sstruct import calcsize, unpack, unpack2
class TFM:

    def __init__(self, file):
        self._read(file)

    def __repr__(self):
        return f'<TFM for {self.family} in {self.codingscheme} at {self.designsize:g}pt>'

    def _read(self, file):
        if hasattr(file, 'read'):
            data = file.read()
        else:
            with open(file, 'rb') as fp:
                data = fp.read()
        self._data = data
        if len(data) < SIZES_SIZE:
            raise TFMException('Too short input file')
        sizes = SimpleNamespace()
        unpack2(SIZES_FORMAT, data, sizes)
        if sizes.lf < 0:
            raise TFMException('The file claims to have negative or zero length!')
        if len(data) < sizes.lf * 4:
            raise TFMException('The file has fewer bytes than it claims!')
        for name, length in vars(sizes).items():
            if length < 0:
                raise TFMException("The subfile size: '{name}' is negative!")
        if sizes.lh < 2:
            raise TFMException(f'The header length is only {sizes.lh}!')
        if sizes.bc > sizes.ec + 1 or sizes.ec > 255:
            raise TFMException(f'The character code range {sizes.bc}..{sizes.ec} is illegal!')
        if sizes.nw == 0 or sizes.nh == 0 or sizes.nd == 0 or (sizes.ni == 0):
            raise TFMException('Incomplete subfiles for character dimensions!')
        if sizes.ne > 256:
            raise TFMException(f'There are {ne} extensible recipes!')
        if sizes.lf != 6 + sizes.lh + (sizes.ec - sizes.bc + 1) + sizes.nw + sizes.nh + sizes.nd + sizes.ni + sizes.nl + sizes.nk + sizes.ne + sizes.np:
            raise TFMException('Subfile sizes don’t add up to the stated total')
        char_base = 6 + sizes.lh - sizes.bc
        width_base = char_base + sizes.ec + 1
        height_base = width_base + sizes.nw
        depth_base = height_base + sizes.nh
        italic_base = depth_base + sizes.nd
        lig_kern_base = italic_base + sizes.ni
        kern_base = lig_kern_base + sizes.nl
        exten_base = kern_base + sizes.nk
        param_base = exten_base + sizes.ne

        def char_info(c):
            return 4 * (char_base + c)

        def width_index(c):
            return data[char_info(c)]

        def noneexistent(c):
            return c < sizes.bc or c > sizes.ec or width_index(c) == 0

        def height_index(c):
            return data[char_info(c) + 1] // 16

        def depth_index(c):
            return data[char_info(c) + 1] % 16

        def italic_index(c):
            return data[char_info(c) + 2] // 4

        def tag(c):
            return data[char_info(c) + 2] % 4

        def remainder(c):
            return data[char_info(c) + 3]

        def width(c):
            r = 4 * (width_base + width_index(c))
            return read_fixed(r, 'v')['v']

        def height(c):
            r = 4 * (height_base + height_index(c))
            return read_fixed(r, 'v')['v']

        def depth(c):
            r = 4 * (depth_base + depth_index(c))
            return read_fixed(r, 'v')['v']

        def italic(c):
            r = 4 * (italic_base + italic_index(c))
            return read_fixed(r, 'v')['v']

        def exten(c):
            return 4 * (exten_base + remainder(c))

        def lig_step(i):
            return 4 * (lig_kern_base + i)

        def lig_kern_command(i):
            command = SimpleNamespace()
            unpack2(LIG_KERN_COMMAND, data[i:], command)
            return command

        def kern(i):
            r = 4 * (kern_base + i)
            return read_fixed(r, 'v')['v']

        def param(i):
            return 4 * (param_base + i)

        def read_fixed(index, key, obj=None):
            ret = unpack2(f'>;{key}:{FIXED_FORMAT}', data[index:], obj)
            return ret[0]
        unpack(HEADER_FORMAT4, [0] * HEADER_SIZE4, self)
        offset = 24
        length = sizes.lh * 4
        self.extraheader = {}
        if length >= HEADER_SIZE4:
            rest = unpack2(HEADER_FORMAT4, data[offset:], self)[1]
            if self.face < 18:
                s = self.face % 2
                b = self.face // 2
                self.face = 'MBL'[b % 3] + 'RI'[s] + 'RCE'[b // 3]
            for i in range(sizes.lh - HEADER_SIZE4 // 4):
                rest = unpack2(f'>;HEADER{i + 18}:l', rest, self.extraheader)[1]
        elif length >= HEADER_SIZE3:
            unpack2(HEADER_FORMAT3, data[offset:], self)
        elif length >= HEADER_SIZE2:
            unpack2(HEADER_FORMAT2, data[offset:], self)
        elif length >= HEADER_SIZE1:
            unpack2(HEADER_FORMAT1, data[offset:], self)
        self.fonttype = VANILLA
        scheme = self.codingscheme.upper()
        if scheme.startswith('TEX MATH SY'):
            self.fonttype = MATHSY
        elif scheme.startswith('TEX MATH EX'):
            self.fonttype = MATHEX
        self.fontdimens = {}
        for i in range(sizes.np):
            name = f'PARAMETER{i + 1}'
            if i <= 6:
                name = BASE_PARAMS[i]
            elif self.fonttype == MATHSY and i <= 21:
                name = MATHSY_PARAMS[i - 7]
            elif self.fonttype == MATHEX and i <= 12:
                name = MATHEX_PARAMS[i - 7]
            read_fixed(param(i), name, self.fontdimens)
        lig_kern_map = {}
        self.right_boundary_char = None
        self.left_boundary_char = None
        if sizes.nl > 0:
            cmd = lig_kern_command(lig_step(0))
            if cmd.skip_byte == 255:
                self.right_boundary_char = cmd.next_char
            cmd = lig_kern_command(lig_step(sizes.nl - 1))
            if cmd.skip_byte == 255:
                self.left_boundary_char = 256
                r = 256 * cmd.op_byte + cmd.remainder
                lig_kern_map[self.left_boundary_char] = r
        self.chars = {}
        for c in range(sizes.bc, sizes.ec + 1):
            if width_index(c) > 0:
                self.chars[c] = info = {}
                info['width'] = width(c)
                if height_index(c) > 0:
                    info['height'] = height(c)
                if depth_index(c) > 0:
                    info['depth'] = depth(c)
                if italic_index(c) > 0:
                    info['italic'] = italic(c)
                char_tag = tag(c)
                if char_tag == NO_TAG:
                    pass
                elif char_tag == LIG_TAG:
                    lig_kern_map[c] = remainder(c)
                elif char_tag == LIST_TAG:
                    info['nextlarger'] = remainder(c)
                elif char_tag == EXT_TAG:
                    info['varchar'] = varchar = {}
                    for i in range(4):
                        part = data[exten(c) + i]
                        if i == 3 or part > 0:
                            name = 'rep'
                            if i == 0:
                                name = 'top'
                            elif i == 1:
                                name = 'mid'
                            elif i == 2:
                                name = 'bot'
                            if noneexistent(part):
                                varchar[name] = c
                            else:
                                varchar[name] = part
        self.ligatures = {}
        self.kerning = {}
        for c, i in sorted(lig_kern_map.items()):
            cmd = lig_kern_command(lig_step(i))
            if cmd.skip_byte > STOP_FLAG:
                i = 256 * cmd.op_byte + cmd.remainder
            while i < sizes.nl:
                cmd = lig_kern_command(lig_step(i))
                if cmd.skip_byte > STOP_FLAG:
                    pass
                elif cmd.op_byte >= KERN_FLAG:
                    r = 256 * (cmd.op_byte - KERN_FLAG) + cmd.remainder
                    self.kerning.setdefault(c, {})[cmd.next_char] = kern(r)
                else:
                    r = cmd.op_byte
                    if r == 4 or (r > 7 and r != 11):
                        lig = r
                    else:
                        lig = ''
                        if r % 4 > 1:
                            lig += '/'
                        lig += 'LIG'
                        if r % 2 != 0:
                            lig += '/'
                        while r > 3:
                            lig += '>'
                            r -= 4
                    self.ligatures.setdefault(c, {})[cmd.next_char] = (lig, cmd.remainder)
                if cmd.skip_byte >= STOP_FLAG:
                    break
                i += cmd.skip_byte + 1