import pickle
def ShowDescriptors(self):
    """ prints out a list of the descriptors

    """
    if self.simpleList is None:
        raise NotImplementedError('Need to have a simpleList defined')
    print('#---------')
    print('Simple:')
    for desc in self.simpleList:
        print(desc)
    if self.compoundList:
        print('#---------')
        print('Compound:')
        for desc in self.compoundList:
            print(desc)