import logging
class BinarySet(object):

    @staticmethod
    def get_interval():
        return (0, 1, 1)

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_integer():
        return True

    @staticmethod
    def is_binary():
        return True