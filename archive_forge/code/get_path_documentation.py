from __future__ import absolute_import, division, print_function
Get the value of a path within an object

    :param var: The var from which the value is retrieved
    :type var: should be dict or list, but jinja can sort that out
    :param path: The path to get
    :type path: should be a string but jinja can sort that out
    :param environment: The jinja Environment
    :type environment: Environment
    :return: The result of the jinja evaluation
    :rtype: any
    