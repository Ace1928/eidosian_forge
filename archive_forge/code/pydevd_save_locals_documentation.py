import sys

    Factory for the 'save_locals_impl' method. This may seem like a complicated pattern but it is essential that the method is created at
    module load time. Inner imports after module load time would cause an occasional debugger deadlock due to the importer lock and debugger
    lock being taken in different order in  different threads.
    