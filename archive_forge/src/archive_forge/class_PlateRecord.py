import warnings
import json
import csv
import numpy as np
from Bio import BiopythonParserWarning
class PlateRecord:
    """PlateRecord object for storing Phenotype Microarray plates data.

    A PlateRecord stores all the wells of a particular phenotype
    Microarray plate, along with metadata (if any). The single wells can be
    accessed calling their id as an index or iterating on the PlateRecord:

    >>> from Bio import phenotype
    >>> plate = phenotype.read("phenotype/Plate.json", "pm-json")
    >>> well = plate['A05']
    >>> for well in plate:
    ...    print(well.id)
    ...
    A01
    ...

    The plate rows and columns can be queried with an indexing system similar
    to NumPy and other matrices:

    >>> print(plate[1])
    Plate ID: PM01
    Well: 12
    Rows: 1
    Columns: 12
    PlateRecord('WellRecord['B01'], WellRecord['B02'], WellRecord['B03'], ..., WellRecord['B12']')

    >>> print(plate[:,1])
    Plate ID: PM01
    Well: 8
    Rows: 8
    Columns: 1
    PlateRecord('WellRecord['A02'], WellRecord['B02'], WellRecord['C02'], ..., WellRecord['H02']')

    Single WellRecord objects can be accessed using this indexing system:

    >>> print(plate[1,2])
    Plate ID: PM01
    Well ID: B03
    Time points: 384
    Minum signal 0.00 at time 11.00
    Maximum signal 76.25 at time 18.00
    WellRecord('(0.0, 11.0), (0.25, 11.0), (0.5, 11.0), (0.75, 11.0), (1.0, 11.0), ..., (95.75, 11.0)')

    The presence of a particular well can be inspected with the "in" keyword:
    >>> 'A01' in plate
    True

    All the wells belonging to a "row" (identified by the first character of
    the well id) in the plate can be obtained:

    >>> for well in plate.get_row('H'):
    ...     print(well.id)
    ...
    H01
    H02
    H03
    ...

    All the wells belonging to a "column" (identified by the number of the well)
    in the plate can be obtained:

    >>> for well in plate.get_column(12):
    ...     print(well.id)
    ...
    A12
    B12
    C12
    ...

    Two PlateRecord objects can be compared: if all their wells are equal the
    two plates are considered equal:

    >>> plate2 = phenotype.read("phenotype/Plate.json", "pm-json")
    >>> plate == plate2
    True

    Two PlateRecord object can be summed up or subtracted from each other: the
    the signals of each well will be summed up or subtracted. The id of the
    left operand will be kept:

    >>> plate3 = plate + plate2
    >>> print(plate3.id)
    PM01

    Many Phenotype Microarray plate have a "negative control" well, which can
    be subtracted to all wells:

    >>> subplate = plate.subtract_control()

    """

    def __init__(self, plateid, wells=None):
        """Initialize the class."""
        self.id = plateid
        if wells is None:
            wells = []
        self.qualifiers = {}
        self._wells = {}
        try:
            for w in wells:
                self._is_well(w)
                self[w.id] = w
        except TypeError:
            raise TypeError('You must provide an iterator-like object containing the single wells')
        self._update()

    def _update(self):
        """Update the rows and columns string identifiers (PRIVATE)."""
        self._rows = sorted({x[0] for x in self._wells})
        self._columns = sorted({x[1:] for x in self._wells})

    def _is_well(self, obj):
        """Check if the given object is a WellRecord object (PRIVATE).

        Used both for the class constructor and the __setitem__ method
        """
        if not isinstance(obj, WellRecord):
            raise ValueError(f'A WellRecord type object is needed as value (got {type(obj)})')

    def __getitem__(self, index):
        """Access part of the plate.

        Depending on the indices, you can get a WellRecord object
        (representing a single well of the plate),
        or another plate
        (representing some part or all of the original plate).

        plate[wid] gives a WellRecord (if wid is a WellRecord id)
        plate[r,c] gives a WellRecord
        plate[r] gives a row as a PlateRecord
        plate[r,:] gives a row as a PlateRecord
        plate[:,c] gives a column as a PlateRecord

        plate[:] and plate[:,:] give a copy of the plate

        Anything else gives a subset of the original plate, e.g.
        plate[0:2] or plate[0:2,:] uses only row 0 and 1
        plate[:,1:3] uses only columns 1 and 2
        plate[0:2,1:3] uses only rows 0 & 1 and only cols 1 & 2

        >>> from Bio import phenotype
        >>> plate = phenotype.read("phenotype/Plate.json", "pm-json")

        You can access a well of the plate, using its id.

        >>> w = plate['A01']

        You can access a row of the plate as a PlateRecord using an integer
        index:

        >>> first_row = plate[0]
        >>> print(first_row)
        Plate ID: PM01
        Well: 12
        Rows: 1
        Columns: 12
        PlateRecord('WellRecord['A01'], WellRecord['A02'], WellRecord['A03'], ..., WellRecord['A12']')
        >>> last_row = plate[-1]
        >>> print(last_row)
        Plate ID: PM01
        Well: 12
        Rows: 1
        Columns: 12
        PlateRecord('WellRecord['H01'], WellRecord['H02'], WellRecord['H03'], ..., WellRecord['H12']')

        You can also access use python's slice notation to sub-plates
        containing only some of the plate rows:

        >>> sub_plate = plate[2:5]
        >>> print(sub_plate)
        Plate ID: PM01
        Well: 36
        Rows: 3
        Columns: 12
        PlateRecord('WellRecord['C01'], WellRecord['C02'], WellRecord['C03'], ..., WellRecord['E12']')

        This includes support for a step, i.e. plate[start:end:step], which
        can be used to select every second row:

        >>> sub_plate = plate[::2]

        You can also use two indices to specify both rows and columns.
        Using simple integers gives you the single wells. e.g.

        >>> w = plate[3, 4]
        >>> print(w.id)
        D05

        To get a single column use this syntax:

        >>> sub_plate = plate[:, 4]
        >>> print(sub_plate)
        Plate ID: PM01
        Well: 8
        Rows: 8
        Columns: 1
        PlateRecord('WellRecord['A05'], WellRecord['B05'], WellRecord['C05'], ..., WellRecord['H05']')

        Or, to get part of a column,

        >>> sub_plate = plate[1:3, 4]
        >>> print(sub_plate)
        Plate ID: PM01
        Well: 2
        Rows: 2
        Columns: 1
        PlateRecord(WellRecord['B05'], WellRecord['C05'])

        However, in general you get a sub-plate,

        >>> print(plate[1:5, 3:6])
        Plate ID: PM01
        Well: 12
        Rows: 4
        Columns: 3
        PlateRecord('WellRecord['B04'], WellRecord['B05'], WellRecord['B06'], ..., WellRecord['E06']')

        This should all seem familiar to anyone who has used the NumPy
        array or matrix objects.
        """
        if isinstance(index, str):
            try:
                return self._wells[index]
            except KeyError:
                raise KeyError(f'Well {index} not found!')
        elif isinstance(index, int):
            try:
                row = self._rows[index]
            except IndexError:
                raise IndexError('Row %d not found!' % index)
            return PlateRecord(self.id, filter(lambda x: x.id.startswith(row), self._wells.values()))
        elif isinstance(index, slice):
            rows = self._rows[index]
            return PlateRecord(self.id, filter(lambda x: x.id[0] in rows, self._wells.values()))
        elif len(index) != 2:
            raise TypeError('Invalid index type.')
        row_index, col_index = index
        if isinstance(row_index, int) and isinstance(col_index, int):
            try:
                row = self._rows[row_index]
            except IndexError:
                raise IndexError('Row %d not found!' % row_index)
            try:
                col = self._columns[col_index]
            except IndexError:
                raise IndexError('Column %d not found!' % col_index)
            return self._wells[row + col]
        elif isinstance(row_index, int):
            try:
                row = self._rows[row_index]
            except IndexError:
                raise IndexError('Row %d not found!' % row_index)
            cols = self._columns[col_index]
            return PlateRecord(self.id, filter(lambda x: x.id.startswith(row) and x.id[1:] in cols, self._wells.values()))
        elif isinstance(col_index, int):
            try:
                col = self._columns[col_index]
            except IndexError:
                raise IndexError('Columns %d not found!' % col_index)
            rows = self._rows[row_index]
            return PlateRecord(self.id, filter(lambda x: x.id.endswith(col) and x.id[0] in rows, self._wells.values()))
        else:
            rows = self._rows[row_index]
            cols = self._columns[col_index]
            return PlateRecord(self.id, filter(lambda x: x.id[0] in rows and x.id[1:] in cols, self._wells.values()))

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError('Well identifier should be string-like')
        self._is_well(value)
        if value.id != key:
            raise ValueError("WellRecord ID and provided key are different (got '%s' and '%s')" % (type(value.id), type(key)))
        self._wells[key] = value
        self._update()

    def __delitem__(self, key):
        if not isinstance(key, str):
            raise ValueError('Well identifier should be string-like')
        del self._wells[key]
        self._update()

    def __iter__(self):
        for well in sorted(self._wells):
            yield self._wells[well]

    def __contains__(self, wellid):
        if wellid in self._wells:
            return True
        return False

    def __len__(self):
        """Return the number of wells in this plate."""
        return len(self._wells)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._wells == other._wells
        else:
            return False

    def __add__(self, plate):
        """Add another PlateRecord object.

        The wells in both plates must be the same

        A new PlateRecord object is returned, having the same id as the
        left operand.
        """
        if not isinstance(plate, PlateRecord):
            raise TypeError('Expecting a PlateRecord object')
        if {x.id for x in self} != {x.id for x in plate}:
            raise ValueError('The two plates have different wells')
        wells = []
        for w in self:
            wells.append(w + plate[w.id])
        newp = PlateRecord(self.id, wells=wells)
        return newp

    def __sub__(self, plate):
        """Subtract another PlateRecord object.

        The wells in both plates must be the same

        A new PlateRecord object is returned, having the same id as the
        left operand.
        """
        if not isinstance(plate, PlateRecord):
            raise TypeError('Expecting a PlateRecord object')
        if {x.id for x in self} != {x.id for x in plate}:
            raise ValueError('The two plates have different wells')
        wells = []
        for w in self:
            wells.append(w - plate[w.id])
        newp = PlateRecord(self.id, wells=wells)
        return newp

    def get_row(self, row):
        """Get all the wells of a given row.

        A row is identified with a letter (e.g. 'A')
        """
        try:
            row = str(row)
        except Exception:
            raise ValueError('Row identifier should be string-like')
        if len(row) > 1:
            raise ValueError('Row identifier must be of maximum one letter')
        for w in sorted(filter(lambda x: x.startswith(row), self._wells)):
            yield self._wells[w]

    def get_column(self, column):
        """Get all the wells of a given column.

        A column is identified with a number (e.g. '6')
        """
        try:
            column = int(column)
        except Exception:
            raise ValueError('Column identifier should be a number')
        for w in sorted(filter(lambda x: x.endswith('%02d' % column), self._wells)):
            yield self._wells[w]

    def subtract_control(self, control='A01', wells=None):
        """Subtract a 'control' well from the other plates wells.

        By default the control is subtracted to all wells, unless
        a list of well ID is provided

        The control well should belong to the plate
        A new PlateRecord object is returned
        """
        if control not in self:
            raise ValueError('Control well not present in plate')
        wcontrol = self[control]
        if wells is None:
            wells = self._wells.keys()
        missing = {w for w in wells if w not in self}
        if missing:
            raise ValueError('Some wells to be subtracted are not present')
        nwells = []
        for w in self:
            if w.id in wells:
                nwells.append(w - wcontrol)
            else:
                nwells.append(w)
        newp = PlateRecord(self.id, wells=nwells)
        return newp

    def __repr__(self):
        """Return a (truncated) representation of the plate for debugging."""
        if len(self._wells) > 4:
            return "%s('%s, ..., %s')" % (self.__class__.__name__, ', '.join(["%s['%s']" % (self[x].__class__.__name__, self[x].id) for x in sorted(self._wells.keys())[:3]]), "%s['%s']" % (self[sorted(self._wells.keys())[-1]].__class__.__name__, self[sorted(self._wells.keys())[-1]].id))
        else:
            return '%s(%s)' % (self.__class__.__name__, ', '.join(["%s['%s']" % (self[x].__class__.__name__, self[x].id) for x in sorted(self._wells.keys())]))

    def __str__(self):
        """Return a human readable summary of the record (string).

        The python built in function str works by calling the object's __str__
        method.  e.g.

        >>> from Bio import phenotype
        >>> record = next(phenotype.parse("phenotype/Plates.csv", "pm-csv"))
        >>> print(record)
        Plate ID: PM01
        Well: 96
        Rows: 8
        Columns: 12
        PlateRecord('WellRecord['A01'], WellRecord['A02'], WellRecord['A03'], ..., WellRecord['H12']')

        Note that long well lists are shown truncated.
        """
        lines = []
        if self.id:
            lines.append(f'Plate ID: {self.id}')
        lines.append('Well: %i' % len(self))
        lines.append('Rows: %d' % len({x.id[0] for x in self}))
        lines.append('Columns: %d' % len({x.id[1:3] for x in self}))
        lines.append(repr(self))
        return '\n'.join(lines)