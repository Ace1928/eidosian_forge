from sys import version_info as _swig_python_version_info
import weakref
class IntVarIterator(BaseObject):
    """
     The class Iterator has two direct subclasses. HoleIterators
     iterates over all holes, that is value removed between the
     current min and max of the variable since the last time the
     variable was processed in the queue. DomainIterators iterates
     over all elements of the variable domain. Both iterators are not
     robust to domain changes. Hole iterators can also report values outside
     the current min and max of the variable.
     HoleIterators should only be called from a demon attached to the
     variable that has created this iterator.
     IntVar* current_var;
     std::unique_ptr<IntVarIterator> it(current_var->MakeHoleIterator(false));
     for (const int64_t hole : InitAndGetValues(it)) {
    use the hole
     }
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined - class is abstract')
    __repr__ = _swig_repr

    def Init(self):
        """ This method must be called before each loop."""
        return _pywrapcp.IntVarIterator_Init(self)

    def Ok(self):
        """ This method indicates if we can call Value() or not."""
        return _pywrapcp.IntVarIterator_Ok(self)

    def Value(self):
        """ This method returns the current value of the iterator."""
        return _pywrapcp.IntVarIterator_Value(self)

    def Next(self):
        """ This method moves the iterator to the next value."""
        return _pywrapcp.IntVarIterator_Next(self)

    def DebugString(self):
        """ Pretty Print."""
        return _pywrapcp.IntVarIterator_DebugString(self)

    def __iter__(self):
        self.Init()
        return self

    def next(self):
        if self.Ok():
            result = self.Value()
            self.Next()
            return result
        else:
            raise StopIteration()

    def __next__(self):
        return self.next()