from reportlab.graphics.barcode.common import Barcode
class ECC200DataMatrix(Barcode):
    """This code only supports a Type 12 (44x44) C40 encoded data matrix.
    This is the size and encoding that Royal Mail wants on all mail from October 1st 2015.
    see https://bitbucket.org/rptlab/reportlab/issues/69/implementations-of-code-128-auto-and-data
    """
    barWidth = 4

    def __init__(self, *args, **kwargs):
        Barcode.__init__(self, *args, **kwargs)
        self.row_modules = 44
        self.col_modules = 44
        self.row_regions = 2
        self.col_regions = 2
        self.cw_data = 144
        self.cw_ecc = 56
        self.row_usable_modules = self.row_modules - self.row_regions * 2
        self.col_usable_modules = self.col_modules - self.col_regions * 2

    def validate(self):
        self.valid = 1
        for c in self.value:
            if ord(c) > 255:
                self.valid = 0
                break
        else:
            self.validated = self.value

    def _encode_c40_char(self, char):
        o = ord(char)
        encoded = []
        if o == 32 or (o >= 48 and o <= 57) or (o >= 65 and o <= 90):
            if o == 32:
                encoded.append(o - 29)
            elif o >= 48 and o <= 57:
                encoded.append(o - 44)
            else:
                encoded.append(o - 51)
        elif o >= 0 and o <= 31:
            encoded.append(0)
            encoded.append(o)
        elif o >= 33 and o <= 64 or (o >= 91 and o <= 95):
            encoded.append(1)
            if o >= 33 and o <= 64:
                encoded.append(o - 33)
            else:
                encoded.append(o - 69)
        elif o >= 96 and o <= 127:
            encoded.append(2)
            encoded.append(o - 96)
        elif o >= 128 and o <= 255:
            encoded.append(1)
            encoded.append(30)
            encoded += self._encode_c40_char(chr(o - 128))
        else:
            raise Exception('Cannot encode %s (%s)' % (char, o))
        return encoded

    def _encode_c40(self, value):
        encoded = []
        for c in value:
            encoded += self._encode_c40_char(c)
        while len(encoded) % 3:
            encoded.append(0)
        codewords = []
        codewords.append(230)
        for i in range(0, len(encoded), 3):
            chunk = encoded[i:i + 3]
            total = chunk[0] * 1600 + chunk[1] * 40 + chunk[2] + 1
            codewords.append(total // 256)
            codewords.append(total % 256)
        codewords.append(254)
        if len(codewords) > self.cw_data:
            raise Exception('Too much data to fit into a data matrix of this size')
        if len(codewords) < self.cw_data:
            codewords.append(129)
            while len(codewords) < self.cw_data:
                r = 149 * (len(codewords) + 1) % 253 + 1
                codewords.append((129 + r) % 254)
        return codewords

    def _gfsum(self, int1, int2):
        return int1 ^ int2

    def _gfproduct(self, int1, int2):
        if int1 == 0 or int2 == 0:
            return 0
        else:
            return ALOGVAL[(LOGVAL[int1] + LOGVAL[int2]) % 255]

    def _get_reed_solomon_code(self, data, num_code_words):
        """
        This method is basically verbatim from "huBarcode" which is BSD licensed
        https://github.com/hudora/huBarcode/blob/master/hubarcode/datamatrix/reedsolomon.py
        """
        cw_factors = FACTORS[num_code_words]
        code_words = [0] * num_code_words
        for data_word in data:
            tmp = self._gfsum(data_word, code_words[-1])
            for j in range(num_code_words - 1, -1, -1):
                code_words[j] = self._gfproduct(tmp, cw_factors[j])
                if j > 0:
                    code_words[j] = self._gfsum(code_words[j - 1], code_words[j])
        code_words.reverse()
        return code_words

    def _get_next_bits(self, data):
        value = data.pop(0)
        bits = []
        for i in range(0, 8):
            bits.append(value >> i & 1)
        bits.reverse()
        return bits

    def _place_bit(self, row, col, bit):
        if row < 0:
            row += self.row_usable_modules
            col += 4 - (self.row_usable_modules + 4) % 8
        if col < 0:
            col += self.col_usable_modules
            row += 4 - (self.col_usable_modules + 4) % 8
        self._matrix[row][col] = bit

    def _place_bit_corner_1(self, data):
        bits = self._get_next_bits(data)
        self._place_bit(self.row_usable_modules - 1, 0, bits[0])
        self._place_bit(self.row_usable_modules - 1, 1, bits[1])
        self._place_bit(self.row_usable_modules - 1, 2, bits[2])
        self._place_bit(0, self.col_usable_modules - 2, bits[3])
        self._place_bit(0, self.col_usable_modules - 1, bits[4])
        self._place_bit(1, self.col_usable_modules - 1, bits[5])
        self._place_bit(2, self.col_usable_modules - 1, bits[6])
        self._place_bit(3, self.col_usable_modules - 1, bits[7])

    def _place_bit_corner_2(self, data):
        bits = self._get_next_bits(data)
        self._place_bit(self.row_usable_modules - 3, 0, bits[0])
        self._place_bit(self.row_usable_modules - 2, 0, bits[1])
        self._place_bit(self.row_usable_modules - 1, 0, bits[2])
        self._place_bit(0, self.col_usable_modules - 4, bits[3])
        self._place_bit(0, self.col_usable_modules - 3, bits[4])
        self._place_bit(0, self.col_usable_modules - 2, bits[5])
        self._place_bit(0, self.col_usable_modules - 1, bits[6])
        self._place_bit(1, self.col_usable_modules - 1, bits[7])

    def _place_bit_corner_3(self, data):
        bits = self._get_next_bits(data)
        self._place_bit(self.row_usable_modules - 3, 0, bits[0])
        self._place_bit(self.row_usable_modules - 2, 0, bits[1])
        self._place_bit(self.row_usable_modules - 1, 0, bits[2])
        self._place_bit(0, self.col_usable_modules - 2, bits[3])
        self._place_bit(0, self.col_usable_modules - 1, bits[4])
        self._place_bit(1, self.col_usable_modules - 1, bits[5])
        self._place_bit(2, self.col_usable_modules - 1, bits[6])
        self._place_bit(3, self.col_usable_modules - 1, bits[7])

    def _place_bit_corner_4(self, data):
        bits = self._get_next_bits(data)
        self._place_bit(self.row_usable_modules - 1, 0, bits[0])
        self._place_bit(self.row_usable_modules - 1, self.col_usable_modules - 1, bits[1])
        self._place_bit(0, self.col_usable_modules - 3, bits[2])
        self._place_bit(0, self.col_usable_modules - 2, bits[3])
        self._place_bit(0, self.col_usable_modules - 1, bits[4])
        self._place_bit(1, self.col_usable_modules - 3, bits[5])
        self._place_bit(1, self.col_usable_modules - 2, bits[6])
        self._place_bit(1, self.col_usable_modules - 1, bits[7])

    def _place_bit_standard(self, data, row, col):
        bits = self._get_next_bits(data)
        self._place_bit(row - 2, col - 2, bits[0])
        self._place_bit(row - 2, col - 1, bits[1])
        self._place_bit(row - 1, col - 2, bits[2])
        self._place_bit(row - 1, col - 1, bits[3])
        self._place_bit(row - 1, col, bits[4])
        self._place_bit(row, col - 2, bits[5])
        self._place_bit(row, col - 1, bits[6])
        self._place_bit(row, col, bits[7])

    def _create_matrix(self, data):
        """
        This method is heavily influenced by "huBarcode" which is BSD licensed
        https://github.com/hudora/huBarcode/blob/master/hubarcode/datamatrix/placement.py
        """
        rows = self.row_usable_modules
        cols = self.col_usable_modules
        self._matrix = self._create_empty_matrix(rows, cols)
        row = 4
        col = 0
        while True:
            if row == rows and col == 0:
                self._place_bit_corner_1(data)
            elif row == rows - 2 and col == 0 and cols % 4:
                self._place_bit_corner_2(data)
            elif row == rows - 2 and col == 0 and (cols % 8 == 4):
                self._place_bit_corner_3(data)
            elif row == rows + 4 and col == 2 and (cols % 8 == 0):
                self._place_bit_corner_4(data)
            while True:
                if row < rows and col >= 0 and (self._matrix[row][col] is None):
                    self._place_bit_standard(data, row, col)
                row -= 2
                col += 2
                if row < 0 or col >= cols:
                    break
            row += 1
            col += 3
            while True:
                if row >= 0 and col < cols and (self._matrix[row][col] is None):
                    self._place_bit_standard(data, row, col)
                row += 2
                col -= 2
                if row >= rows or col < 0:
                    break
            row += 3
            col += 1
            if row >= rows and col >= cols:
                break
        for row in self._matrix:
            for i in range(0, cols):
                if row[i] is None:
                    row[i] = 0
        return self._matrix

    def _create_data_regions(self, matrix):
        regions = []
        col_offset = 0
        row_offset = 0
        rows = int(self.row_usable_modules / self.row_regions)
        cols = int(self.col_usable_modules / self.col_regions)
        while col_offset < self.row_regions:
            while row_offset < self.col_regions:
                r_offset = col_offset * rows
                c_offset = row_offset * cols
                region = matrix[r_offset:rows + r_offset]
                for i in range(0, len(region)):
                    region[i] = region[i][c_offset:cols + c_offset]
                regions.append(region)
                row_offset += 1
            row_offset = 0
            col_offset += 1
        return regions

    def _create_empty_matrix(self, row, col):
        matrix = []
        for i in range(0, row):
            matrix.append([None] * col)
        return matrix

    def _wrap_data_regions_with_finders(self, regions):
        wrapped = []
        for region in regions:
            matrix = self._create_empty_matrix(int(self.col_modules / self.col_regions), int(self.row_modules / self.row_regions))
            for i, rows in enumerate(region):
                for j, data in enumerate(rows):
                    matrix[i + 1][j + 1] = data
            for i, row in enumerate(matrix):
                if i == 0:
                    for j, col in enumerate(row):
                        row[j] = (j + 1) % 2
                elif i + 1 == len(matrix):
                    for j, col in enumerate(row):
                        row[j] = 1
                else:
                    row[0] = 1
                    row[-1] = i % 2
            wrapped.append(matrix)
        return wrapped

    def _merge_data_regions(self, regions):
        merged = []
        for i in range(0, len(regions), self.row_regions):
            chunk = regions[i:i + self.row_regions]
            j = 0
            while j < len(chunk[0]):
                merged_row = []
                for row in chunk:
                    merged_row += row[j]
                merged.append(merged_row)
                j += 1
        return merged

    def encode(self):
        if hasattr(self, 'encoded'):
            return self.encoded
        encoded = self._encode_c40(self.validated)
        encoded += self._get_reed_solomon_code(encoded, self.cw_ecc)
        matrix = self._create_matrix(encoded)
        data_regions = self._create_data_regions(matrix)
        wrapped = self._wrap_data_regions_with_finders(data_regions)
        self.encoded = self._merge_data_regions(wrapped)
        self.encoded.reverse()
        return self.encoded

    def computeSize(self, *args):
        self._height = self.row_modules * self.barWidth
        self._width = self.col_modules * self.barWidth

    def draw(self):
        for y, row in enumerate(self.encoded):
            for x, data in enumerate(row):
                if data:
                    self.rect(self.x + x * self.barWidth, self.y + y * self.barWidth, self.barWidth, self.barWidth)