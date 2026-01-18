import pygame

# Initialize Pygame
pygame.init()

# Set up the display
width = 800
height = 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive Shape Drawing")

# Colors in RGB format
colors = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (255, 255, 0)
}
current_color = colors['red']

# Shape options
shapes = ['circle', 'rectangle', 'line']
current_shape = shapes[0]

# Variables to track drawing
drawing = False
start_pos = None
end_pos = None

# Lists to store drawn shapes
circles = []
rectangles = []
lines = []

# Main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            start_pos = pygame.mouse.get_pos()
            end_pos = start_pos
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                end_pos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            # Add the shape to the respective list
            if current_shape == 'circle':
                center = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) 
// 2)
                radius = int(((end_pos[0] - start_pos[0])**2 + (end_pos[1] - 
start_pos[1])**2)**0.5)
                circles.append((center, radius, current_color))
            elif current_shape == 'rectangle':
                x = min(start_pos[0], end_pos[0])
                y = min(start_pos[1], end_pos[1])
                width = abs(end_pos[0] - start_pos[0])
                height = abs(end_pos[1] - start_pos[1])
                rectangles.append((x, y, width, height, current_color))
            elif current_shape == 'line':
                lines.append((start_pos, end_pos, current_color))
        elif event.type == pygame.KEYDOWN:
            # Change color
            if event.key == pygame.K_1:
                current_color = colors['red']
            elif event.key == pygame.K_2:
                current_color = colors['blue']
            elif event.key == pygame.K_3:
                current_color = colors['green']
            elif event.key == pygame.K_4:
                current_color = colors['yellow']
            # Change shape
            elif event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_c:
                circles.clear()
                rectangles.clear()
                lines.clear()
            elif event.key == pygame.K_TAB:
                idx = shapes.index(current_shape)
                current_shape = shapes[(idx + 1) % len(shapes)]

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw all circles
    for circle in circles:
        pygame.draw.circle(screen, circle[2], circle[0], circle[1])

    # Draw all rectangles
    for rect in rectangles:
        pygame.draw.rect(screen, rect[4], (rect[0], rect[1], rect[2], rect[3]))

    # Draw all lines
    for line in lines:
        pygame.draw.line(screen, line[2], line[0], line[1], 5)

    # If drawing, preview the current shape
    if drawing:
        if current_shape == 'circle':
            center = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 
2)
            radius = int(((end_pos[0] - start_pos[0])**2 + (end_pos[1] - 
start_pos[1])**2)**0.5)
            pygame.draw.circle(screen, current_color, center, radius, 1)
        elif current_shape == 'rectangle':
            x = min(start_pos[0], end_pos[0])
            y = min(start_pos[1], end_pos[1])
            width = abs(end_pos[0] - start_pos[0])
            height = abs(end_pos[1] - start_pos[1])
            pygame.draw.rect(screen, current_color, (x, y, width, height), 1)
        elif current_shape == 'line':
            pygame.draw.line(screen, current_color, start_pos, end_pos, 5)

    # Update display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
