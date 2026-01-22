

     >>> obj = LazySig(lambda x:x,10)
     >>> obj[1]
     1
     >>> obj[-1]
     9
     >>> try:
     ...   obj[10]
     ... except IndexError:
     ...   1
     ... else:
     ...   0
     1
     >>> try:
     ...   obj[-10]
     ... except IndexError:
     ...   1
     ... else:
     ...   0
     1

    